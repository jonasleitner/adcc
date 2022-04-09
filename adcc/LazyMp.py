#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
import libadcc
import numpy as np

from .functions import direct_sum, evaluate, einsum
from .misc import cached_property, cached_member_function
from .ReferenceState import ReferenceState
from .OneParticleOperator import OneParticleOperator, product_trace
from .Intermediates import register_as_intermediate
from .timings import Timer, timed_member_call
from .MoSpaces import split_spaces
from . import block as b


class LazyMp:
    def __init__(self, hf, first_order_singles=False):
        """
        Initialise the class dealing with the M/oller-Plesset ground state.
        """
        if isinstance(hf, libadcc.HartreeFockSolution_i):
            hf = ReferenceState(hf)
        if not isinstance(hf, ReferenceState):
            raise TypeError("hf needs to be a ReferenceState "
                            "or a HartreeFockSolution_i")
        self.reference_state = hf
        self.mospaces = hf.mospaces
        self.timer = Timer()
        self.has_core_occupied_space = hf.has_core_occupied_space
        self.first_order_singles = first_order_singles

    def __getattr__(self, attr):
        # Shortcut some quantities, which are needed most often
        if attr.startswith("t2") and len(attr) == 4:  # t2oo, t2oc, t2cc
            xxvv = b.__getattr__(attr[2:4] + "vv")
            return self.t2(xxvv)
        else:
            raise AttributeError

    @cached_member_function
    def df(self, space):
        """Delta Fock matrix"""
        hf = self.reference_state
        s1, s2 = split_spaces(space)
        fC = hf.fock(s1 + s1).diagonal()
        fv = hf.fock(s2 + s2).diagonal()
        return direct_sum("-i+a->ia", fC, fv)

    @cached_member_function
    def t2_hyl(self, space):
        """
        Iterative T2 amplitudes through minimization
        of the Hylleraas functional
        """

        if space != b.oovv:
            raise NotImplementedError("T2 hylleraas amplitudes not implemented "
                                      f"for space {space}")
        print(f"\nComputing iterative T2 amplitudes for space {space}")
        hf = self.reference_state

        # guess setup
        t2_amp = hf.eri(space)

        maxiter = 100
        conv_tol = 1e-15
        print("iteration, residue norm Doubles")
        for i in range(maxiter):
            # sum_c(t_ijac f_cb - t_ijbc f_ca) = 2 * sum_c t_ijac f_cb
            # - sum_k(t_ikab f_jk - t_jkab f_ik) = - 2 * sum_k t_ikab f_jk
            residue = (
                + 2.0 * einsum("ijac,cb->ijab", t2_amp, hf.fvv)
                - 2.0 * einsum("ikab,jk->ijab", t2_amp, hf.foo)
                - hf.eri(space)
            )
            residue = residue.antisymmetrise((0, 1)).antisymmetrise((2, 3))

            if residue.select_n_absmax(1)[0][1] > 1e3:
                print("max value of residue to large")
                print(residue.select_n_absmax(3))
                exit()
            # update t2 amplitudes
            t2_amp -= 0.25 * residue

            # compute the norm of the residue
            norm = np.sqrt(einsum("ijab,ijab->", residue, residue))
            print(f"{i+1}         {norm}")
            if norm < conv_tol:
                print("Iterative T2 amplitudes converged!")
                break
            elif norm > 1e3:
                print("diverged :(")
                exit()

        # compare to canonical t2_amplitudes:
        diff = t2_amp - self.t2(space)
        diff_norm = np.sqrt(einsum('ijab,ijab->', diff, diff))
        print("diff Hyl-RSPT amps: norm = ", diff_norm)
        print("diff Hyl-RSPT amps: max val = ", diff.select_n_absmax(3))
        return t2_amp

    @cached_member_function
    def t2_with_singles(self, space):
        """iterative first order T amplitudes (including singles) through
           minimization of the Hylleraas functional
           """

        if space != b.oovv:
            raise NotImplementedError("T2 hylleraas amplitudes not implemented "
                                      f"for space {space}")
        print("\nComputing iterative T amplitudes (including singles) for space",
              space)
        hf = self.reference_state

        # guess setup
        td_amp = hf.eri(space)
        ts_amp = OneParticleOperator(self.mospaces, is_symmetric=True)
        ts_amp = ts_amp.ov.ones_like()

        maxiter = 100
        conv_tol = 1e-15
        print("iteration, residue norm:   Singles   Doubles")
        for i in range(maxiter):
            # total residue (with singles):
            doubles_r = (
                + einsum('ia,jb->ijab', ts_amp, hf.fov)
                - 0.25 * hf.eri(space)
                - 0.5 * einsum('ikab,jk->ijab', td_amp, hf.foo)
                + 0.5 * einsum('ijac,cb->ijab', td_amp, hf.fvv)
            )
            doubles_r = doubles_r.antisymmetrise((0, 1)).antisymmetrise((2, 3))

            if doubles_r.select_n_absmax(1)[0][1] > 1e3:
                print("max value of residue to large:")
                print(doubles_r.select_n_absmax(3))
                exit()

            singles_r = - einsum('ja,ij->ia', ts_amp, hf.foo) + \
                einsum('ib,ba->ia', ts_amp, hf.fvv) + \
                einsum('ijab,bj->ia', td_amp, hf.fvo)

            if singles_r.select_n_absmax(1)[0][1] > 1e3:
                print("max value of singles residue to large:")
                print(singles_r.select_n_absmax(3))
                exit()

            # add residue to amplutides
            # here different scaling factor than without singles, because prefactors
            # in the residue expression are different
            ts_amp -= 0.5 * singles_r
            td_amp -= 1.0 * doubles_r

            # compute the norm of the residues
            norm_d = np.sqrt(einsum("ijab,ijab->", doubles_r, doubles_r))
            norm_s = np.sqrt(einsum('ia,ia->', singles_r, singles_r))
            print(f"{i+1}         {norm_d} / {norm_s}")
            if np.sqrt(norm_d**2 + norm_s**2) < conv_tol:
                break
            elif np.sqrt(norm_d**2 + norm_s**2) > 1e3:
                print("Hylleraas optimization including singles diverged.")
                print(f"Singles norm: {norm_s}. Doubles norm: {norm_d}")
                exit()

        # compare doubles amplitudes to canonical ones:
        diff = td_amp - self.t2(space)
        diff_norm = np.sqrt(einsum('ijab,ijab->', diff, diff))
        print("diff Hyl(with S)-RSPT amps: norm = ", diff_norm)
        print("diff Hyl(with S)-RSPT amps: max val = ", diff.select_n_absmax(3))
        print("converged singles amplitudes:\n", ts_amp)
        # hacking the singles amplitudes in the function cache.
        if 'ts1_hyl' not in self._function_cache:
            self._function_cache['ts1_hyl'] = {}
        self._function_cache['ts1_hyl'][b.ov] = ts_amp
        return td_amp

    @cached_member_function
    def ts1_hyl(self, space):
        if space != b.ov:
            raise NotImplementedError("T^S_1 term not implemented "
                                      f"for space {space}.")
        self.t2_with_singles(b.oovv)
        return self._function_cache['ts1_hyl'][space]

    @cached_member_function
    def t2(self, space):
        """T2 amplitudes"""
        print("\nComputing canonical T2 amplitudes")
        hf = self.reference_state
        sp = split_spaces(space)
        assert all(s == b.v for s in sp[2:])
        eia = self.df(sp[0] + b.v)
        ejb = self.df(sp[1] + b.v)
        return (
            hf.eri(space) / direct_sum("ia+jb->ijab", eia, ejb).symmetrise((2, 3))
        )

    @cached_member_function
    def ts2_hyl(self, space):
        """Second order iterative singles amplitudes without first order
           single amplitudes."""
        if space != b.ov:
            raise NotImplementedError("T^S_2 term not implemented "
                                      f"for space {space}.")

        hf = self.reference_state
        # first order amplitudes from minimizing E(2)
        td1 = self.t2_hyl(b.oovv)

        print("Computing iterative 2nd order ground state wavefunction "
              "by minimizing the fourth order energy.")

        # guess setup (zeros for singles, triples, quadruples)
        ts2 = OneParticleOperator(self.mospaces).ov.zeros_like()
        td2 = td1.copy()

        # switch to numpy
        td1_np = td1.to_ndarray()
        ts2_np = ts2.to_ndarray()
        td2_np = td2.to_ndarray()
        o = self.mospaces.n_orbs('o1')
        v = self.mospaces.n_orbs('v1')
        tt2_np = np.zeros((o, o, o, v, v, v))
        tq2_np = np.zeros((o, o, o, o, v, v, v, v))

        # Currently the following expansion is assumed below:
        # |Psi^(1)> = + t1s |s> - t1d |d>
        # |Psi^(2)> = + t2s |s> + t2d |d> + t2t |t> + t2q |q>

        # terms that do not depend on the second order amplitudes.
        # Constant, because the first oder amplitudes are kept fixed.
        # <2|(H1 - E1)|1>:
        i_s_np = (  # changed sign of all terms (wrt to sympy result)
            - 0.5 * np.einsum('icab,ijab->jc', hf.ovvv.to_ndarray(), td1_np)
            - 0.5 * np.einsum('ijka,ijab->kb', hf.ooov.to_ndarray(), td1_np)
        )
        i_d_np = (  # changed sign of all terms (wrt to sympy result)
            + np.einsum('ibja,ikac->jkbc', hf.ovov.to_ndarray(), td1_np)
            - 1 / 8 * np.einsum('cdab,ijab->ijcd', hf.vvvv.to_ndarray(), td1_np)
            - 1 / 8 * np.einsum('ijkl,ijab->klab', hf.oooo.to_ndarray(), td1_np)
        )
        i_t_np = (  # changed sign of all terms (wrt to sympy result)
            - 0.25 * np.einsum('jkia,ilbc->jklabc', hf.ooov.to_ndarray(), td1_np)
            - 0.25 * np.einsum('iabc,jkad->ijkbcd', hf.ovvv.to_ndarray(), td1_np)
        )
        i_q_np = (  # changed sign of all terms (wrt to sympy result)
            - 1 / 16 * np.einsum(
                'ijab,klcd->ijklabcd', hf.oovv.to_ndarray(), td1_np
            )
        )

        maxiter = 500
        conv_tol = 1e-15
        spaces = ["Singles", "Doubles", "Triples", "Quadruples"]
        print("Residue norm:     ", "               ".join(spaces))
        for i in range(maxiter):
            # switch to np
            singles_r = (
                + i_s_np + np.einsum('ba,ia->ib', hf.fvv.to_ndarray(), ts2_np)
                + np.einsum('ia,ijab->jb', hf.fov.to_ndarray(), td2_np)
                - np.einsum('ij,ia->ja', hf.foo.to_ndarray(), ts2_np)
            )
            doubles_r = (
                + i_d_np
                + np.einsum('ai,jb->ijab', hf.fvo.to_ndarray(), ts2_np)
                + 0.5 * np.einsum('ba,ijac->ijbc', hf.fvv.to_ndarray(), td2_np)
                + 0.25 * np.einsum('ia,ijkabc->jkbc', hf.fov.to_ndarray(), tt2_np)
                - 0.5 * np.einsum('ij,ikab->jkab', hf.foo.to_ndarray(), td2_np)
            )
            doubles_r = 0.5 * (doubles_r - np.moveaxis(doubles_r, 0, 1))
            doubles_r = 0.5 * (doubles_r - np.moveaxis(doubles_r, 2, 3))
            triples_r = (
                + i_t_np
                + 0.25 * np.einsum('ai,jkbc->ijkabc', hf.fvo.to_ndarray(), td2_np)
                + 1 / 12 * np.einsum(
                    'ba,ijkacd->ijkbcd', hf.fvv.to_ndarray(), tt2_np
                )
                + 1 / 36 * np.einsum(
                    'ia,ijklabcd->jklbcd', hf.fov.to_ndarray(), tq2_np
                )
                - 1 / 12 * np.einsum(
                    'ij,iklabc->jklabc', hf.foo.to_ndarray(), tt2_np
                )
            )
            # Need to set up the whole symmetry of bra/ket at once!!

            # occ: (1 - P_ij - P_jk)(1 - P_ik)
            triples_r = 1 / 6 * (
                triples_r - np.moveaxis(triples_r, 0, 1)  # jik
                - np.moveaxis(triples_r, 1, 2)  # ikj
                - np.moveaxis(triples_r, [0, 2], [2, 0])  # kji
                + np.moveaxis(triples_r, [0, 1, 2], [2, 0, 1])  # jki
                + np.moveaxis(triples_r, [0, 1, 2], [1, 2, 0])  # kij
            )
            # virt: (1 - P_ab - P_bc)(1 - P_ac)
            triples_r = 1 / 6 * (
                triples_r - np.moveaxis(triples_r, 3, 4)  # bac
                - np.moveaxis(triples_r, 4, 5)  # acb
                - np.moveaxis(triples_r, [3, 5], [5, 3])  # cba
                + np.moveaxis(triples_r, [3, 4, 5], [5, 3, 4])  # bca
                + np.moveaxis(triples_r, [3, 4, 5], [4, 5, 3])  # cab
            )

            quadruples_r = (
                + i_q_np
                + 1 / 36 * np.einsum(
                    'ai,jklbcd->ijklabcd', hf.fvo.to_ndarray(), tt2_np
                )
                + 1 / 144 * np.einsum(
                    'ba,ijklacde->ijklbcde', hf.fvv.to_ndarray(), tq2_np
                )
                - 1 / 144 * np.einsum(
                    'ij, iklmabcd->jklmabcd', hf.foo.to_ndarray(), tq2_np
                )
            )
            # occ: (1 - P_ij - P_ik - P_il)(1 - P_jk - P_jl)(1 - P_kl)
            quadruples_r = 1 / 24 * (
                quadruples_r - np.moveaxis(quadruples_r, 2, 3)  # ijlk
                - np.moveaxis(quadruples_r, 1, 2)  # ikjl
                + np.moveaxis(quadruples_r, [1, 2, 3], [3, 1, 2])  # iklj
                - np.moveaxis(quadruples_r, [1, 3], [3, 1])  # ilkj
                + np.moveaxis(quadruples_r, [1, 2, 3], [2, 3, 1])  # iljk
                - np.moveaxis(quadruples_r, 0, 1)  # jikl
                + np.moveaxis(quadruples_r, [0, 2], [1, 3])  # jilk
                + np.moveaxis(quadruples_r, [0, 1, 2], [2, 0, 1])  # jkil
                - np.moveaxis(quadruples_r, [0, 1, 2, 3], [3, 0, 1, 2])  # jkli
                + np.moveaxis(quadruples_r, [0, 1, 2, 3], [3, 0, 2, 1])  # jlki
                - np.moveaxis(quadruples_r, [0, 1, 2, 3], [2, 0, 3, 1])  # jlik
                - np.moveaxis(quadruples_r, [0, 2], [2, 0])  # kjil
                + np.moveaxis(quadruples_r, [0, 1, 2, 3], [3, 1, 0, 2])  # kjli
                + np.moveaxis(quadruples_r, [0, 1, 2, 3], [1, 2, 0, 3])  # kijl
                - np.moveaxis(quadruples_r, [0, 1, 2, 3], [1, 3, 0, 2])  # kilj
                + np.moveaxis(quadruples_r, [0, 1, 2, 3], [2, 3, 0, 1])  # klij
                - np.moveaxis(quadruples_r, [0, 1, 2, 3], [3, 2, 0, 1])  # klji
                - np.moveaxis(quadruples_r, [0, 3], [3, 0])  # ljki
                + np.moveaxis(quadruples_r, [0, 1, 2, 3], [2, 1, 3, 0])  # ljik
                + np.moveaxis(quadruples_r, [0, 1, 2, 3], [3, 2, 1, 0])  # lkji
                - np.moveaxis(quadruples_r, [0, 1, 2, 3], [2, 3, 1, 0])  # lkij
                + np.moveaxis(quadruples_r, [0, 1, 2, 3], [1, 3, 2, 0])  # likj
                - np.moveaxis(quadruples_r, [0, 1, 2, 3], [1, 2, 3, 0])  # lijk
            )
            # virt: (1 - P_ab - P_ac - P_ad)(1 - P_bc - P_bd)(1 - P_cd)
            quadruples_r = 1 / 24 * (
                quadruples_r - np.moveaxis(quadruples_r, 6, 7)  # abdc
                - np.moveaxis(quadruples_r, 5, 6)  # acbd
                + np.moveaxis(quadruples_r, [5, 6, 7], [7, 5, 6])  # acdb
                - np.moveaxis(quadruples_r, [5, 7], [7, 5])  # adcb
                + np.moveaxis(quadruples_r, [5, 6, 7], [6, 7, 5])  # adbc
                - np.moveaxis(quadruples_r, 4, 5)  # bacd
                + np.moveaxis(quadruples_r, [4, 6], [5, 7])  # badc
                + np.moveaxis(quadruples_r, [4, 5, 6], [6, 4, 5])  # bcad
                - np.moveaxis(quadruples_r, [4, 5, 6, 7], [7, 4, 5, 6])  # bcda
                + np.moveaxis(quadruples_r, [4, 5, 6, 7], [7, 4, 6, 5])  # bdca
                - np.moveaxis(quadruples_r, [4, 5, 6, 7], [6, 4, 7, 5])  # bdac
                - np.moveaxis(quadruples_r, [4, 6], [6, 4])  # cbad
                + np.moveaxis(quadruples_r, [4, 5, 6, 7], [7, 5, 4, 6])  # cbda
                + np.moveaxis(quadruples_r, [4, 5, 6, 7], [5, 6, 4, 7])  # cabd
                - np.moveaxis(quadruples_r, [4, 5, 6, 7], [5, 7, 4, 6])  # cadb
                + np.moveaxis(quadruples_r, [4, 5, 6, 7], [6, 7, 4, 5])  # cdab
                - np.moveaxis(quadruples_r, [4, 5, 6, 7], [7, 6, 4, 5])  # cdba
                - np.moveaxis(quadruples_r, [4, 7], [7, 4])  # dbca
                + np.moveaxis(quadruples_r, [4, 5, 6, 7], [6, 5, 7, 4])  # dbac
                + np.moveaxis(quadruples_r, [4, 5, 6, 7], [7, 6, 5, 4])  # dcba
                - np.moveaxis(quadruples_r, [4, 5, 6, 7], [6, 7, 5, 4])  # dcab
                + np.moveaxis(quadruples_r, [4, 5, 6, 7], [5, 7, 6, 4])  # dacb
                - np.moveaxis(quadruples_r, [4, 5, 6, 7], [5, 6, 7, 4])  # dabc
            )

            # equation from michael wormit:
            # canonical Hylleraas equation multiplied by factor 4!
            # doubles_r = 0.25 * (
            #     # sum_c + f_ac t2_ijcb + f_bc t2_ijac = +2 * sum_c f_bc t2_ijac
            #     + 2 * einsum('bc,ijac->ijab', hf.fvv, td2)
            #     # sum_c - f_ik t2_kjab - f_jk t2_ikab = -2 * sum_c f_jk t2_ikab
            #     - 2 * einsum('jk,ikab->ijab', hf.foo, td2)
            #     # const terms:
            #     + 0.5 * einsum('abcd,ijcd->ijab', hf.vvvv, td1)
            #     + 0.5 * einsum('klij,klab->ijab', hf.oooo, td1)
            #     # sum_kc t_ikac <kb||jc> - t_jkac <kb||ic>
            #     #      + t_jkbc <ka||ic> - t_ikbc <ka||jc>
            #     #      = 4 * sum_kc t_ikac <kb||jc>
            #     - 4 * einsum('kbjc,ikac->ijab', hf.ovov, td1)
            # )
            # if singles_r.select_n_absmax(1)[0][1] > 1e3:
            #     print("max value of T^S_2 residue too large.")
            #     print(singles_r.select_n_absmax(3))
            #     exit()
            # if doubles_r.select_n_absmax(1)[0][1] > 1e3:
            #     print("max value of T^D_2 residue too large.")
            #     print(doubles_r.select_n_absmax(3))
            #     exit()

            # update
            ts2_np -= 0.5 * singles_r
            td2_np -= 1.0 * doubles_r
            tt2_np -= 8.0 * triples_r
            tq2_np -= 111.0 * quadruples_r

            # check norm of the residuals
            norm_s = np.sqrt(np.einsum('ia,ia->', singles_r, singles_r))
            norm_d = np.sqrt(np.einsum('ijab,ijab->', doubles_r, doubles_r))
            norm_t = np.sqrt(np.einsum('ijkabc,ijkabc->', triples_r, triples_r))
            norm_q = np.sqrt(np.einsum(
                'ijklabcd,ijklabcd->', quadruples_r, quadruples_r)
            )
            print(f"{i+1}          {norm_s} / {norm_d} / {norm_t} / {norm_q}")
            if np.sqrt(norm_s ** 2 + norm_d ** 2 + norm_t ** 2 + norm_q ** 2) \
                    < conv_tol:
                print("Converged!")
                break
            elif any([norm > 1e3 for norm in
                     [norm_s ** 2, norm_d ** 2, norm_t ** 2, norm_q ** 2]]):
                print("Second order wavefunction diverged.")
                break

        ts2_can = self.mp2_diffdm.ov
        dif = ts2_np - ts2_can.to_ndarray()
        dif_norm = np.sqrt(np.einsum('ia,ia->', dif, dif))
        print("difference to canonical singles amplitudes:")
        print(f"norm: {dif_norm}")
        print("max difference: ", np.amax(dif))

        td2_can = self.td2(b.oovv)
        dif = td2_np + td2_can.to_ndarray()  # different sign apparently
        dif_norm = np.sqrt(np.einsum('ijab,ijab->', dif, dif))
        print("difference to canonical doubles amplitudes:")
        print(f"norm: {dif_norm}")
        print("max difference: ", np.amax(dif))

        tt2_can = self.tt2(b.ooovvv)
        dif = tt2_np - tt2_can
        dif_norm = np.sqrt(np.einsum('ijkabc,ijkabc->', dif, dif))
        print("difference to canonical triples amplitudes:\nnorm: ", dif_norm)
        print("max difference: ", np.amax(dif))

        tq2_can = self.tq2(b.oooovvvv)
        dif = tq2_np - tq2_can
        dif_norm = np.sqrt(np.einsum('ijklabcd,ijklabcd->', dif, dif))
        print("difference to canonical quadruple amplitudes:\nnorm: ", dif_norm)
        print("max difference: ", np.amax(dif))
        return OneParticleOperator(self.mospaces).ov.set_from_ndarray(ts2_np, 1e-15)

    @cached_member_function
    def ts2_with_singles(self, space):
        """Second order iterative singles amplitudes with first order
           single amplitudes."""
        if space != b.ov:
            raise NotImplementedError("T^S_2 term not implemented "
                                      f"for space {space}.")

        hf = self.reference_state
        # first order amplitudes from minimizing E(2)
        ts1 = self.ts1_hyl(space)
        td1 = self.t2_with_singles(b.oovv)

        print("Computing iterative 2nd order ground state wavefunction "
              "with first order singles "
              "by minimizing the fourth order energy.")

        # guess setup (zeros for Triples and Quadruples)
        ts2 = ts1.copy()
        td2 = td1.copy()

        # switch to numpy
        ts1_np = ts1.to_ndarray()
        td1_np = td1.to_ndarray()
        ts2_np = ts2.to_ndarray()
        td2_np = td2.to_ndarray()
        o = self.mospaces.n_orbs('o1')
        v = self.mospaces.n_orbs('v1')
        tt2_np = np.zeros((o, o, o, v, v, v))
        tq2_np = np.zeros((o, o, o, o, v, v, v, v))

        # Currently the following expansion is assumed below:
        # |Psi^(1)> = - t1s |s> - t1d |d>
        # |Psi^(2)> = + t2s |s> + t2d |d> + t2t |t> + t2q |q>

        # terms that do not depend on the second order amplitudes.
        # Constant, because the first oder amplitudes are kept fixed.
        # <2|(H1 - E1)|1>:
        i_s_np = (  # changed sign of all terms (wrt to sympy result)
            + np.einsum('ibja,ia->jb', hf.ovov.to_ndarray(), ts1_np)
            - 0.5 * np.einsum('icab,ijab->jc', hf.ovvv.to_ndarray(), td1_np)
            - 0.5 * np.einsum('ijka,ijab->kb', hf.ooov.to_ndarray(), td1_np)
        )
        i_d_np = (  # changed sign of all terms (wrt to sympy result)
            - 0.5 * np.einsum('jkia,ib->jkab', hf.ooov.to_ndarray(), ts1_np)
            - 0.5 * np.einsum('iabc,ja->ijbc', hf.ovvv.to_ndarray(), ts1_np)
            + np.einsum('ibja,ikac->jkbc', hf.ovov.to_ndarray(), td1_np)
            - 1 / 8 * np.einsum('cdab,ijab->ijcd', hf.vvvv.to_ndarray(), td1_np)
            - 1 / 8 * np.einsum('ijkl,ijab->klab', hf.oooo.to_ndarray(), td1_np)
        )
        i_t_np = (  # changed sign of all terms (wrt to sympy result)
            - 0.25 * np.einsum('ijab,kc->ijkabc', hf.oovv.to_ndarray(), ts1_np)
            - 0.25 * np.einsum('jkia,ilbc->jklabc', hf.ooov.to_ndarray(), td1_np)
            - 0.25 * np.einsum('iabc,jkad->ijkbcd', hf.ovvv.to_ndarray(), td1_np)
        )
        i_q_np = (  # changed sign of all terms (wrt to sympy result)
            - 1 / 16 * np.einsum(
                'ijab,klcd->ijklabcd', hf.oovv.to_ndarray(), td1_np
            )
        )

        maxiter = 500
        conv_tol = 1e-15
        spaces = ["Singles", "Doubles", "Triples", "Quadruples"]
        print("Residue norm:     ", "               ".join(spaces))
        for i in range(maxiter):
            # switch to np
            singles_r = (
                + i_s_np + np.einsum('ba,ia->ib', hf.fvv.to_ndarray(), ts2_np)
                + np.einsum('ia,ijab->jb', hf.fov.to_ndarray(), td2_np)
                - np.einsum('ij,ia->ja', hf.foo.to_ndarray(), ts2_np)
            )
            doubles_r = (
                + i_d_np
                + np.einsum('ai,jb->ijab', hf.fvo.to_ndarray(), ts2_np)
                + 0.5 * np.einsum('ba,ijac->ijbc', hf.fvv.to_ndarray(), td2_np)
                + 0.25 * np.einsum('ia,ijkabc->jkbc', hf.fov.to_ndarray(), tt2_np)
                - 0.5 * np.einsum('ij,ikab->jkab', hf.foo.to_ndarray(), td2_np)
            )
            doubles_r = 0.5 * (doubles_r - np.moveaxis(doubles_r, 0, 1))
            doubles_r = 0.5 * (doubles_r - np.moveaxis(doubles_r, 2, 3))
            triples_r = (
                + i_t_np
                + 0.25 * np.einsum('ai,jkbc->ijkabc', hf.fvo.to_ndarray(), td2_np)
                + 1 / 12 * np.einsum(
                    'ba,ijkacd->ijkbcd', hf.fvv.to_ndarray(), tt2_np
                )
                + 1 / 36 * np.einsum(
                    'ia,ijklabcd->jklbcd', hf.fov.to_ndarray(), tq2_np
                )
                - 1 / 12 * np.einsum(
                    'ij,iklabc->jklabc', hf.foo.to_ndarray(), tt2_np
                )
            )
            # Need to set up the whole symmetry of bra/ket at once!!

            # occ: (1 - P_ij - P_jk)(1 - P_ik)
            triples_r = 1 / 6 * (
                triples_r - np.moveaxis(triples_r, 0, 1)  # jik
                - np.moveaxis(triples_r, 1, 2)  # ikj
                - np.moveaxis(triples_r, [0, 2], [2, 0])  # kji
                + np.moveaxis(triples_r, [0, 1, 2], [2, 0, 1])  # jki
                + np.moveaxis(triples_r, [0, 1, 2], [1, 2, 0])  # kij
            )
            # virt: (1 - P_ab - P_bc)(1 - P_ac)
            triples_r = 1 / 6 * (
                triples_r - np.moveaxis(triples_r, 3, 4)  # bac
                - np.moveaxis(triples_r, 4, 5)  # acb
                - np.moveaxis(triples_r, [3, 5], [5, 3])  # cba
                + np.moveaxis(triples_r, [3, 4, 5], [5, 3, 4])  # bca
                + np.moveaxis(triples_r, [3, 4, 5], [4, 5, 3])  # cab
            )

            quadruples_r = (
                + i_q_np
                + 1 / 36 * np.einsum(
                    'ai,jklbcd->ijklabcd', hf.fvo.to_ndarray(), tt2_np
                )
                + 1 / 144 * np.einsum(
                    'ba,ijklacde->ijklbcde', hf.fvv.to_ndarray(), tq2_np
                )
                - 1 / 144 * np.einsum(
                    'ij, iklmabcd->jklmabcd', hf.foo.to_ndarray(), tq2_np
                )
            )
            # occ: (1 - P_ij - P_ik - P_il)(1 - P_jk - P_jl)(1 - P_kl)
            quadruples_r = 1 / 24 * (
                quadruples_r - np.moveaxis(quadruples_r, 2, 3)  # ijlk
                - np.moveaxis(quadruples_r, 1, 2)  # ikjl
                + np.moveaxis(quadruples_r, [1, 2, 3], [3, 1, 2])  # iklj
                - np.moveaxis(quadruples_r, [1, 3], [3, 1])  # ilkj
                + np.moveaxis(quadruples_r, [1, 2, 3], [2, 3, 1])  # iljk
                - np.moveaxis(quadruples_r, 0, 1)  # jikl
                + np.moveaxis(quadruples_r, [0, 2], [1, 3])  # jilk
                + np.moveaxis(quadruples_r, [0, 1, 2], [2, 0, 1])  # jkil
                - np.moveaxis(quadruples_r, [0, 1, 2, 3], [3, 0, 1, 2])  # jkli
                + np.moveaxis(quadruples_r, [0, 1, 2, 3], [3, 0, 2, 1])  # jlki
                - np.moveaxis(quadruples_r, [0, 1, 2, 3], [2, 0, 3, 1])  # jlik
                - np.moveaxis(quadruples_r, [0, 2], [2, 0])  # kjil
                + np.moveaxis(quadruples_r, [0, 1, 2, 3], [3, 1, 0, 2])  # kjli
                + np.moveaxis(quadruples_r, [0, 1, 2, 3], [1, 2, 0, 3])  # kijl
                - np.moveaxis(quadruples_r, [0, 1, 2, 3], [1, 3, 0, 2])  # kilj
                + np.moveaxis(quadruples_r, [0, 1, 2, 3], [2, 3, 0, 1])  # klij
                - np.moveaxis(quadruples_r, [0, 1, 2, 3], [3, 2, 0, 1])  # klji
                - np.moveaxis(quadruples_r, [0, 3], [3, 0])  # ljki
                + np.moveaxis(quadruples_r, [0, 1, 2, 3], [2, 1, 3, 0])  # ljik
                + np.moveaxis(quadruples_r, [0, 1, 2, 3], [3, 2, 1, 0])  # lkji
                - np.moveaxis(quadruples_r, [0, 1, 2, 3], [2, 3, 1, 0])  # lkij
                + np.moveaxis(quadruples_r, [0, 1, 2, 3], [1, 3, 2, 0])  # likj
                - np.moveaxis(quadruples_r, [0, 1, 2, 3], [1, 2, 3, 0])  # lijk
            )
            # virt: (1 - P_ab - P_ac - P_ad)(1 - P_bc - P_bd)(1 - P_cd)
            quadruples_r = 1 / 24 * (
                quadruples_r - np.moveaxis(quadruples_r, 6, 7)  # abdc
                - np.moveaxis(quadruples_r, 5, 6)  # acbd
                + np.moveaxis(quadruples_r, [5, 6, 7], [7, 5, 6])  # acdb
                - np.moveaxis(quadruples_r, [5, 7], [7, 5])  # adcb
                + np.moveaxis(quadruples_r, [5, 6, 7], [6, 7, 5])  # adbc
                - np.moveaxis(quadruples_r, 4, 5)  # bacd
                + np.moveaxis(quadruples_r, [4, 6], [5, 7])  # badc
                + np.moveaxis(quadruples_r, [4, 5, 6], [6, 4, 5])  # bcad
                - np.moveaxis(quadruples_r, [4, 5, 6, 7], [7, 4, 5, 6])  # bcda
                + np.moveaxis(quadruples_r, [4, 5, 6, 7], [7, 4, 6, 5])  # bdca
                - np.moveaxis(quadruples_r, [4, 5, 6, 7], [6, 4, 7, 5])  # bdac
                - np.moveaxis(quadruples_r, [4, 6], [6, 4])  # cbad
                + np.moveaxis(quadruples_r, [4, 5, 6, 7], [7, 5, 4, 6])  # cbda
                + np.moveaxis(quadruples_r, [4, 5, 6, 7], [5, 6, 4, 7])  # cabd
                - np.moveaxis(quadruples_r, [4, 5, 6, 7], [5, 7, 4, 6])  # cadb
                + np.moveaxis(quadruples_r, [4, 5, 6, 7], [6, 7, 4, 5])  # cdab
                - np.moveaxis(quadruples_r, [4, 5, 6, 7], [7, 6, 4, 5])  # cdba
                - np.moveaxis(quadruples_r, [4, 7], [7, 4])  # dbca
                + np.moveaxis(quadruples_r, [4, 5, 6, 7], [6, 5, 7, 4])  # dbac
                + np.moveaxis(quadruples_r, [4, 5, 6, 7], [7, 6, 5, 4])  # dcba
                - np.moveaxis(quadruples_r, [4, 5, 6, 7], [6, 7, 5, 4])  # dcab
                + np.moveaxis(quadruples_r, [4, 5, 6, 7], [5, 7, 6, 4])  # dacb
                - np.moveaxis(quadruples_r, [4, 5, 6, 7], [5, 6, 7, 4])  # dabc
            )

            # update
            ts2_np -= 0.5 * singles_r
            td2_np -= 1.0 * doubles_r
            tt2_np -= 8.0 * triples_r
            tq2_np -= 111.0 * quadruples_r

            # check norm of the residuals
            norm_s = np.sqrt(np.einsum('ia,ia->', singles_r, singles_r))
            norm_d = np.sqrt(np.einsum('ijab,ijab->', doubles_r, doubles_r))
            norm_t = np.sqrt(np.einsum('ijkabc,ijkabc->', triples_r, triples_r))
            norm_q = np.sqrt(np.einsum(
                'ijklabcd,ijklabcd->', quadruples_r, quadruples_r)
            )
            print(f"{i+1}          {norm_s} / {norm_d} / {norm_t} / {norm_q}")
            if np.sqrt(norm_s ** 2 + norm_d ** 2 + norm_t ** 2 + norm_q ** 2) \
                    < conv_tol:
                print("Converged!")
                break
            elif any([norm > 1e3 for norm in (norm_s, norm_d, norm_t, norm_q)]):
                print("Second order wavefunction diverged.")
                break

        ts2_can = self.mp2_diffdm.ov
        dif = ts2_np - ts2_can.to_ndarray()
        dif_norm = np.sqrt(np.einsum('ia,ia->', dif, dif))
        print("difference to canonical singles amplitudes:")
        print(f"norm: {dif_norm}")
        print("max difference: ", np.amax(dif))

        td2_can = self.td2(b.oovv)
        dif = td2_np + td2_can.to_ndarray()  # different sign apparently
        dif_norm = np.sqrt(np.einsum('ijab,ijab->', dif, dif))
        print("difference to canonical doubles amplitudes:")
        print(f"norm: {dif_norm}")
        print("max difference: ", np.amax(dif))

        tt2_can = self.tt2(b.ooovvv)
        dif = tt2_np - tt2_can
        dif_norm = np.sqrt(np.einsum('ijkabc,ijkabc->', dif, dif))
        print("difference to canonical triples amplitudes:\nnorm: ", dif_norm)
        print("max difference: ", np.amax(dif))

        tq2_can = self.tq2(b.oooovvvv)
        dif = tq2_np - tq2_can
        dif_norm = np.sqrt(np.einsum('ijklabcd,ijklabcd->', dif, dif))
        print("difference to canonical quadruple amplitudes:\nnorm: ", dif_norm)
        print("max difference: ", np.amax(dif))
        return OneParticleOperator(self.mospaces).ov.set_from_ndarray(ts2_np, 1e-15)

    @cached_member_function
    def td2(self, space):
        """Return the T^D_2 term"""
        if space != b.oovv:
            raise NotImplementedError("T^D_2 term not implemented "
                                      f"for space {space}.")
        t2erit = self.t2eri(b.oovv, b.ov).transpose((1, 0, 2, 3))
        denom = direct_sum(
            'ia,jb->ijab', self.df(b.ov), self.df(b.ov)
        ).symmetrise(0, 1)
        return (
            + 4.0 * t2erit.antisymmetrise(2, 3).antisymmetrise(0, 1)
            - 0.5 * self.t2eri(b.oovv, b.vv)
            - 0.5 * self.t2eri(b.oovv, b.oo)
        ) / denom

    @cached_member_function
    def t2eri(self, space, contraction):
        """
        Return the T2 tensor with ERI tensor contraction intermediates.
        These are called pi1 to pi7 in libadc.
        """
        hf = self.reference_state
        key = space + contraction
        expressions = {
            # space + contraction
            b.ooov + b.vv: ('ijbc,kabc->ijka', b.ovvv),
            b.ooov + b.ov: ('ilab,lkjb->ijka', b.ooov),
            b.oovv + b.oo: ('klab,ijkl->ijab', b.oooo),
            b.oovv + b.ov: ('jkac,kbic->ijab', b.ovov),
            b.oovv + b.vv: ('ijcd,abcd->ijab', b.vvvv),
            b.ovvv + b.oo: ('jkbc,jkia->iabc', b.ooov),
            b.ovvv + b.ov: ('ijbd,jcad->iabc', b.ovvv),
        }
        if key not in expressions:
            raise NotImplementedError("t2eri intermediate not implemented "
                                      f"for space '{space}' and contraction "
                                      f"'{contraction}'.")
        contraction_str, eri_block = expressions[key]
        return einsum(contraction_str, self.t2oo, hf.eri(eri_block))

    def ts2(self, space):
        """Returns the second order singles amplitudes T^S_2."""
        if space != b.ov:
            raise NotImplementedError("T^S_2 term not implemented ",
                                      f"for space {space}.")
        return self.mp2_diffdm.ov

    def tt2(self, space):
        # numpy because no libtensor triples yet.
        if space != b.ooovvv:
            raise NotImplementedError("T^T_2 term not implemented "
                                      f"for space {space}.")
        hf = self.reference_state
        # hacking direct_sum with numpy
        eia = self.df(b.ov).to_ndarray()
        delta = np.add.outer(eia, eia)
        delta = np.add.outer(delta, eia)  # shape: ovovov
        delta = -1.0 * np.moveaxis(delta, [0, 1, 2, 3, 4, 5], [0, 3, 1, 4, 2, 5])
        # symmetrise denominator?
        # delta = 1 / 6 * (
        #     delta + np.moveaxis(delta, 0, 1) + np.moveaxis(delta, 1, 2)
        #     + np.moveaxis(delta, [0, 2], [2, 0])
        #     + np.moveaxis(delta, [0, 1, 2], [2, 0, 1])
        #     + np.moveaxis(delta, [0, 1, 2], [1, 2, 0])
        # )
        # delta = 1 / 6 * (
        #     delta + np.moveaxis(delta, 3, 4) + np.moveaxis(delta, 4, 5)
        #     + np.moveaxis(delta, [3, 5], [5, 3])
        #     + np.moveaxis(delta, [3, 4, 5], [5, 3, 4])
        #     + np.moveaxis(delta, [3, 4, 5], [4, 5, 3])
        # )

        # did I have a sign error here?? check manu/adrian result again!
        i1 = + np.einsum('kdbc,ijad->ijkabc', hf.ovvv.to_ndarray(),
                         self.t2(b.oovv).to_ndarray())
        # symmetrization of i1: (1 - P_ki - P_kj)(1 - P_ab - P_ac)
        i1 = (
            i1 - np.moveaxis(i1, 3, 4) - np.moveaxis(i1, [3, 5], [5, 3])
            - np.moveaxis(i1, [0, 2], [2, 0])
            + np.moveaxis(i1, [0, 2, 3], [2, 0, 4])
            + np.moveaxis(i1, [0, 2, 3, 5], [2, 0, 5, 3])
            - np.moveaxis(i1, 1, 2) + np.moveaxis(i1, [1, 3], [2, 4])
            + np.moveaxis(i1, [1, 3, 5], [2, 5, 3])
        )

        i2 = + np.einsum('jklc,ilab->ijkabc', hf.ooov.to_ndarray(),
                         self.t2(b.oovv).to_ndarray())
        # symmetrization of i2: (1 - P_ij - P_ik)(1 - P_ca - P_cb)
        i2 = (
            i2 - np.moveaxis(i2, [3, 5], [5, 3]) - np.moveaxis(i2, 4, 5)
            - np.moveaxis(i2, 0, 1) + np.moveaxis(i2, [0, 3, 5], [1, 5, 3])
            + np.moveaxis(i2, [0, 4], [1, 5]) - np.moveaxis(i2, [0, 2], [2, 0])
            + np.moveaxis(i2, [0, 2, 3, 5], [2, 0, 5, 3])
            + np.moveaxis(i2, [0, 2, 4], [2, 0, 5])
        )
        return (i1 + i2) / delta

    def tq2(self, space):
        if space != b.oooovvvv:
            raise NotImplementedError("T^Q_2 term not implemented "
                                      f"for space {space}.")
        hf = self.reference_state
        # build delta i+j+k+l-a-b-c-d
        # hacking direct_sum with numpy
        eia = self.df(b.ov).to_ndarray()
        delta = np.add.outer(eia, eia)
        delta = np.add.outer(delta, eia)
        delta = np.add.outer(delta, eia)  # shape: ovovovov
        delta = -1.0 * np.moveaxis(
            delta, [0, 1, 2, 3, 4, 5, 6, 7], [0, 4, 1, 5, 2, 6, 3, 7]
        )
        # symmetrise the denominator!
        ret = - np.einsum('ijab,klcd->ijklabcd', hf.oovv.to_ndarray(),
                          self.t2(b.oovv).to_ndarray())
        # symmetrization of ret: (36 terms in total)
        # (1 - P_jk - P_jl - P_ik - P_il + P_ik*P_jl) * (1 - P_bc - P_bd - P_ac
        # - P_ad + P_ac*P_bd)
        ret = (
            ret - np.moveaxis(ret, 5, 6)  # ijklacbd  / P_bc
            - np.moveaxis(ret, [5, 7], [7, 5])  # ijkladcb  / P_bd
            - np.moveaxis(ret, [4, 6], [6, 4])  # ijklcbad  / P_ac
            - np.moveaxis(ret, [4, 7], [7, 4])  # ijkldbca  / P_ad
            + np.moveaxis(ret, [4, 5, 6, 7], [6, 7, 4, 5])  # ijklcdab / P_acbd
            - np.moveaxis(ret, 1, 2)  # ikjlabcd  / P_jk
            + np.moveaxis(ret, [1, 5], [2, 6])  # ikjlacbd / P_jkbc
            + np.moveaxis(ret, [1, 5, 7], [2, 7, 5])  # ikjladcb / P_jkbd
            + np.moveaxis(ret, [1, 4, 6], [2, 6, 4])  # ikjlcbad / P_jkac
            + np.moveaxis(ret, [1, 4, 7], [2, 7, 4])  # ikjldbca / P_jkad
            - np.moveaxis(
                ret, [1, 4, 5, 6, 7], [2, 6, 7, 4, 5]  # ikjlcdab /P_jkacbd
            )
            - np.moveaxis(ret, [1, 3], [3, 1])  # ilkjabcd / P_jl
            + np.moveaxis(ret, [1, 3, 5], [3, 1, 6])  # ilkjacbd / P_jlbc
            + np.moveaxis(ret, [1, 3, 5, 7], [3, 1, 7, 5])  # ilkjadcb /P_jlbd
            + np.moveaxis(ret, [1, 3, 4, 6], [3, 1, 6, 4])  # ilkjcbad / P_jlac
            + np.moveaxis(ret, [1, 3, 4, 7], [3, 1, 7, 4])  # ilkjdbca / P_jlad
            - np.moveaxis(  # ilkjcdab / P_jlacbd
                ret, [1, 3, 4, 5, 6, 7], [3, 1, 6, 7, 4, 5]
            )
            - np.moveaxis(ret, [0, 2], [2, 0])  # kjilabcd /P_ik
            + np.moveaxis(ret, [0, 2, 5], [2, 0, 6])  # kjilacbd / P_ikbc
            + np.moveaxis(ret, [0, 2, 5, 7], [2, 0, 7, 5])  # kjiladcb / P_ikbd
            + np.moveaxis(ret, [0, 2, 4, 6], [2, 0, 6, 4])  # kjilcbad / P_ikac
            + np.moveaxis(ret, [0, 2, 4, 7], [2, 0, 7, 4])  # kjildbca / P_ikad
            - np.moveaxis(  # kjilcdab /P_ikacbd
                ret, [0, 2, 4, 5, 6, 7], [2, 0, 6, 7, 4, 5]
            )
            - np.moveaxis(ret, [0, 3], [3, 0])  # ljkiabcd / P_il
            + np.moveaxis(ret, [0, 3, 5], [3, 0, 6])  # ljkiacbd / P_ilbc
            + np.moveaxis(ret, [0, 3, 5, 7], [3, 0, 7, 5])  # ljkiadcb / P_ilbd
            + np.moveaxis(ret, [0, 3, 4, 6], [3, 0, 6, 4])  # ljkicbad / P_ilac
            + np.moveaxis(ret, [0, 3, 4, 7], [3, 0, 7, 4])  # ljkidbca / P_ilad
            - np.moveaxis(  # ljkicdab / P_ilacbd
                ret, [0, 3, 4, 5, 6, 7], [3, 0, 6, 7, 4, 5]
            )
            + np.moveaxis(ret, [0, 1, 2, 3], [2, 3, 0, 1])  # klijabcd / P_ikjl
            - np.moveaxis(  # klijacbd / P_ikjlbc
                ret, [0, 1, 2, 3, 5], [2, 3, 0, 1, 6]
            )
            - np.moveaxis(  # klijadcb / P_ikjlbd
                ret, [0, 1, 2, 3, 5, 7], [2, 3, 0, 1, 7, 5]
            )
            - np.moveaxis(  # klijcbad / P_ikjlac
                ret, [0, 1, 2, 3, 4, 6], [2, 3, 0, 1, 6, 4]
            )
            - np.moveaxis(  # klijdbca / P_ikjlad
                ret, [0, 1, 2, 3, 4, 7], [2, 3, 0, 1, 7, 4]
            )
            + np.moveaxis(  # klijcdab / P_ikjlacbd
                ret, [0, 1, 2, 3, 4, 5, 6, 7], [2, 3, 0, 1, 6, 7, 4, 5]
            )
        )
        return ret / delta

    @cached_property
    @timed_member_call(timer="timer")
    def mp2_diffdm(self):
        """
        Return the MP2 differensce density in the MO basis.
        """
        hf = self.reference_state
        ret = OneParticleOperator(self.mospaces, is_symmetric=True)
        # NOTE: the following 3 blocks are equivalent to the cvs_p0 intermediates
        # defined at the end of this file
        # td = self.t2_hyl('o1o1v1v1')
        ret.oo = -0.5 * einsum("ikab,jkab->ij", self.t2oo, self.t2oo)
        # ret.oo = -0.5 * einsum("ikab,jkab->ij", td, td)
        ret.ov = -0.5 * (
            + einsum("ijbc,jabc->ia", self.t2oo, hf.ovvv)
            + einsum("jkib,jkab->ia", hf.ooov, self.t2oo)
        ) / self.df(b.ov)
        ret.vv = 0.5 * einsum("ijac,ijbc->ab", self.t2oo, self.t2oo)
        # ret.vv = 0.5 * einsum("ijac,ijbc->ab", td, td)

        if self.has_core_occupied_space:
            # additional terms to "revert" CVS for ground state density
            ret.oo += -0.5 * einsum("iLab,jLab->ij", self.t2oc, self.t2oc)
            ret.ov += -0.5 * (
                + einsum("jMib,jMab->ia", hf.ocov, self.t2oc)
                + einsum("iLbc,Labc->ia", self.t2oc, hf.cvvv)
                + einsum("kLib,kLab->ia", hf.ocov, self.t2oc)
                + einsum("iMLb,LMab->ia", hf.occv, self.t2cc)
                - einsum("iLMb,LMab->ia", hf.occv, self.t2cc)
            ) / self.df(b.ov)
            ret.vv += (
                + 0.5 * einsum("IJac,IJbc->ab", self.t2cc, self.t2cc)
                + 1.0 * einsum("kJac,kJbc->ab", self.t2oc, self.t2oc)
            )
            # compute extra CVS blocks
            ret.cc = -0.5 * (
                + einsum("kIab,kJab->IJ", self.t2oc, self.t2oc)
                + einsum('LIab,LJab->IJ', self.t2cc, self.t2cc)
            )
            ret.co = -0.5 * (
                + einsum("kIab,kjab->Ij", self.t2oc, self.t2oo)
                + einsum("ILab,jLab->Ij", self.t2cc, self.t2oc)
            )
            ret.cv = -0.5 * (
                - einsum("jIbc,jabc->Ia", self.t2oc, hf.ovvv)
                + einsum("jkIb,jkab->Ia", hf.oocv, self.t2oo)
                + einsum("jMIb,jMab->Ia", hf.occv, self.t2oc)
                + einsum("ILbc,Labc->Ia", self.t2cc, hf.cvvv)
                + einsum("kLIb,kLab->Ia", hf.occv, self.t2oc)
                + einsum("LMIb,LMab->Ia", hf.cccv, self.t2cc)
            ) / self.df(b.cv)
        ret.reference_state = self.reference_state
        return evaluate(ret)

    def density(self, level=2):
        """
        Return the MP density in the MO basis with all corrections
        up to the specified order of perturbation theory
        """
        if level == 1:
            return self.reference_state.density
        elif level == 2:
            return self.reference_state.density + self.mp2_diffdm
        else:
            raise NotImplementedError("Only densities for level 1 and 2"
                                      " are implemented.")

    def dipole_moment(self, level=2):
        """
        Return the MP dipole moment at the specified level of
        perturbation theory.
        """
        if level == 1:
            return self.reference_state.dipole_moment
        elif level == 2:
            return self.mp2_dipole_moment
        else:
            raise NotImplementedError("Only dipole moments for level 1 and 2"
                                      " are implemented.")

    @cached_member_function
    def energy_correction(self, level=2):
        """Obtain the MP energy correction at a particular level"""
        if level > 3:
            raise NotImplementedError(f"MP({level}) energy correction "
                                      "not implemented.")
        if level < 2:
            return 0.0
        hf = self.reference_state
        is_cvs = self.has_core_occupied_space
        if level == 2 and not is_cvs:
            terms = [(1.0, hf.oovv, self.t2oo)]
        elif level == 2 and is_cvs:
            terms = [(1.0, hf.oovv, self.t2oo),
                     (2.0, hf.ocvv, self.t2oc),
                     (1.0, hf.ccvv, self.t2cc)]
        elif level == 3 and not is_cvs:
            terms = [(1.0, hf.oovv, self.td2(b.oovv))]
        elif level == 3 and is_cvs:
            raise NotImplementedError("CVS-MP3 energy correction not implemented.")
        return sum(
            -0.25 * pref * eri.dot(t2)
            for pref, eri, t2 in terms
        )

    def energy(self, level=2):
        """
        Obtain the total energy (SCF energy plus all corrections)
        at a particular level of perturbation theory.
        """
        if level == 0:
            # Sum of orbital energies ...
            raise NotImplementedError("Total MP(0) energy not implemented.")

        # Accumulator for all energy terms
        energies = [self.reference_state.energy_scf]
        for il in range(2, level + 1):
            energies.append(self.energy_correction(il))
        return sum(energies)

    def energy_hyl(self, order, singles=False):
        """
        Obtain the total energy according to the n-th order Hylleraas
        functional.
        """
        if order == 0:
            raise NotImplementedError("Total MP(0) Hylleraas energy not",
                                      "implemented")

        energies = [self.reference_state.energy_scf]
        for o in range(2, order + 1):
            energies.append(self.energy_correction_hyl(o, singles))
        canonical = self.energy(order)
        print(f"diff to completely canonical total E: {sum(energies)- canonical}")
        return sum(energies)

    @cached_member_function
    def energy_correction_hyl(self, order=2, singles=False):
        """Calculates the MP(n) Hylleraas corretion."""

        if order > 2:
            raise NotImplementedError(f"Hylleraas energy for MP({order}) not",
                                      "implemented")
        if order < 2:
            return 0.0
        if self.has_core_occupied_space:
            raise NotImplementedError(f"Hylleraas correction for MP({order})",
                                      "not implemented")

        hf = self.reference_state

        if not singles:
            td = self.t2_hyl('o1o1v1v1')
            return - 0.5 * einsum('ijab,ijab->', td, hf.oovv) \
                - 0.5 * einsum('ikab,ijab,jk->', td, td, hf.foo) \
                + 0.5 * einsum('ijac,ijab,cb->', td, td, hf.fvv)
        else:
            td = self.t2_with_singles('o1o1v1v1')
            ts = self.ts1_hyl('o1v1')
            i1 = - einsum('ja,ia,ij->', ts, ts, hf.foo)
            i2 = + einsum('ib,ia,ba->', ts, ts, hf.fvv)
            i3 = + 2 * einsum('ia,ijab,jb->', ts, td, hf.fov)
            # i4 = + einsum('ijab,ia,bj->', td, ts, hf.fvo)
            i5 = - 0.5 * einsum('ijab,ijab->', td, hf.oovv)
            i6 = - 0.5 * einsum('ikab,ijab,jk->', td, td, hf.foo)
            i7 = + 0.5 * einsum('ijac,ijab,cb->', td, td, hf.fvv)
            print(i1, i2, i3, i5, i6, i7)
            # print("can merge i3 and i4?: ", i3 - i4)
            return i1 + i2 + i3 + i5 + i6 + i7

    def to_qcvars(self, properties=False, recurse=False, maxlevel=2):
        """
        Return a dictionary with property keys compatible to a Psi4 wavefunction
        or a QCEngine Atomicresults object.
        """
        qcvars = {}
        for level in range(2, maxlevel + 1):
            try:
                mpcorr = self.energy_correction(level)
                qcvars[f"MP{level} CORRELATION ENERGY"] = mpcorr
                qcvars[f"MP{level} TOTAL ENERGY"] = self.energy(level)
            except NotImplementedError:
                pass
            except ValueError:
                pass

        if properties:
            for level in range(2, maxlevel + 1):
                try:
                    qcvars["MP2 DIPOLE"] = self.dipole_moment(level)
                except NotImplementedError:
                    pass

        if recurse:
            qcvars.update(self.reference_state.to_qcvars(properties, recurse))
        return qcvars

    @property
    def mp2_density(self):
        return self.density(2)

    @cached_property
    def mp2_dipole_moment(self):
        refstate = self.reference_state
        dipole_integrals = refstate.operators.electric_dipole
        mp2corr = -np.array([product_trace(comp, self.mp2_diffdm)
                             for comp in dipole_integrals])
        return refstate.dipole_moment + mp2corr


#
# Register cvs_p0 intermediate
#
@register_as_intermediate
def cvs_p0(hf, mp, intermediates):
    # NOTE: equal to mp2_diffdm if CVS applied for the density
    ret = OneParticleOperator(hf.mospaces, is_symmetric=True)
    ret.oo = -0.5 * einsum("ikab,jkab->ij", mp.t2oo, mp.t2oo)
    ret.ov = -0.5 * (+ einsum("ijbc,jabc->ia", mp.t2oo, hf.ovvv)
                     + einsum("jkib,jkab->ia", hf.ooov, mp.t2oo)) / mp.df(b.ov)
    ret.vv = 0.5 * einsum("ijac,ijbc->ab", mp.t2oo, mp.t2oo)
    return ret


# def triple_symmetry_guess(o, v):
#     n = 0.0
#     ret = np.zeros((o, o, o, v, v, v))
#     for i in range(o):
#         for j in range(i + 1, o):
#             for k in range(j + 1, o):
#                 for a in range(v):
#                     for b in range(a + 1, v):
#                         for c in range(b + 1, v):
#                             ret[i, j, k, a, b, c] = +n  # 1
#                             ret[i, j, k, a, c, b] = -n
#                             ret[i, j, k, b, a, c] = -n
#                             ret[i, j, k, b, c, a] = +n
#                             ret[i, j, k, c, b, a] = -n
#                             ret[i, j, k, c, a, b] = +n
#                             ret[i, k, j, a, b, c] = -n  # 2
#                             ret[i, k, j, a, c, b] = +n
#                             ret[i, k, j, b, a, c] = +n
#                             ret[i, k, j, b, c, a] = -n
#                             ret[i, k, j, c, b, a] = +n
#                             ret[i, k, j, c, a, b] = -n
#                             ret[j, i, k, a, b, c] = -n  # 3
#                             ret[j, i, k, a, c, b] = +n
#                             ret[j, i, k, b, a, c] = +n
#                             ret[j, i, k, b, c, a] = -n
#                             ret[j, i, k, c, b, a] = +n
#                             ret[j, i, k, c, a, b] = -n
#                             ret[j, k, i, a, b, c] = +n  # 4
#                             ret[j, k, i, a, c, b] = -n
#                             ret[j, k, i, b, a, c] = -n
#                             ret[j, k, i, b, c, a] = +n
#                             ret[j, k, i, c, b, a] = -n
#                             ret[j, k, i, c, a, b] = +n
#                             ret[k, j, i, a, b, c] = -n  # 5
#                             ret[k, j, i, a, c, b] = +n
#                             ret[k, j, i, b, a, c] = +n
#                             ret[k, j, i, b, c, a] = -n
#                             ret[k, j, i, c, b, a] = +n
#                             ret[k, j, i, c, a, b] = -n
#                             ret[k, i, j, a, b, c] = +n  # 6
#                             ret[k, i, j, a, c, b] = -n
#                             ret[k, i, j, b, a, c] = -n
#                             ret[k, i, j, b, c, a] = +n
#                             ret[k, i, j, c, b, a] = -n
#                             ret[k, i, j, c, a, b] = +n
#     return ret
