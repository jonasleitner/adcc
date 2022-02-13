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
    def __init__(self, hf):
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
        """Second order iterative singles amplitudes"""
        if space != b.ov:
            raise NotImplementedError("T^S_2 term not implemented "
                                      f"for space {space}.")

        hf = self.reference_state
        # first order amplitudes from minimizing E(2)
        ts1 = self.ts1_hyl(space)
        td1 = self.t2_with_singles(b.oovv)

        # guess setup
        ts2 = ts1.copy()
        td2 = td1.copy()

        # some const intermediates (due to fixed first order amplitudes)
        # i_s_nice = (
        #     - einsum('jaib,jb->ia', hf.ovov, ts1)
        #     - 0.5 * einsum('jabc,ijbc->ia', hf.ovvv, td1)
        #     - 0.5 * einsum('jkib,jkab->ia', hf.ooov, td1)
        # )
        # i_d_nice = (
        #     + 0.5 * einsum('ijka,kb->ijab', hf.ooov, ts1)
        #     + 0.5 * einsum('icab,jc->ijab', hf.ovvv, ts1)
        #     - einsum('kbjc,ikac->ijab', hf.ovov, td1)
        #     + 0.125 * einsum('abcd,ijcd->ijab', hf.vvvv, td1)
        #     + 0.125 * einsum('klij,klab->ijab', hf.oooo, td1)
        # )

        # terms that do not depend on the second order amplitudes.
        # Constant, because the first oder amplitudes are kept fixed.
        # <2|(H1 - E1)|1>:
        i_s = (
            - einsum('ibja,ia->jb', hf.ovov, ts1)
            + 0.5 * einsum('icab,ijab->jc', hf.ovvv, td1)
            + 0.5 * einsum('ijka,ijab->kb', hf.ooov, td1)
        )
        i_d = (
            + 0.5 * einsum('jkia,ib->jkab', hf.ooov, ts1)
            + 0.5 * einsum('iabc,ja->ijbc', hf.ovvv, ts1)
            - einsum('ibja,ikac->jkbc', hf.ovov, td1)
            + 1 / 8 * einsum('cdab,ijab->ijcd', hf.vvvv, td1)
            + 1 / 8 * einsum('ijkl,ijab->klab', hf.oooo, td1)
        )
        # i_t = (
        #     + 0.25 * einsum('ijab,kc->ijkabc', hf.oovv, ts1)
        #     + 0.25 * einsum('jkia,ilbc->jklabc', hf.ooov, td1)
        #     + 0.25 * einsum('iabc,jkad->ijkbcd', hf.ovvv, td1)
        # )
        # i_q = (
        #     + 1 / 16 * direct_sum('ijab,klcd->ijklabcd', hf.oovv, td1)
        # )

        maxiter = 200
        conv_tol = 1e-15
        print("Computing iterative 2nd order ground state wavefunction "
              "by minimizing the fourth order energy.")
        print("Residue norm:     Singles    Doubles")
        for i in range(maxiter):
            singles_r = (
                + i_s + einsum('ba,ia->ib', hf.fvv, ts2)
                + einsum('ia,ijab->jb', hf.fov, td2)
                - einsum('ij,ia->ja', hf.foo, ts2)
            )
            doubles_r = (
                + i_d
                # the direct_sum term below does something weird
                # if neglecting this one, the result is correct.
                + direct_sum('ai,jb->ijab', hf.fvo, ts2)
                + 0.5 * einsum('ba,ijac->ijbc', hf.fvv, td2)
                # + 0.25 * einsum('ia,ijkabc->jkbc', hf.fov, tt2)
                - 0.5 * einsum('ij,ikab->jkab', hf.foo, td2)
            )
            # triples_r = (
            #     + i_t + 0.25 * direct_sum('ai,jkbc->ijkabc', hf.fvo, tt2)
            #     + 1 / 12 * einsum('ba,ijkacd->ijkbcd', hf.fvv, tt2)
            #     + 1 / 36 * einsum('ia,ijklabcd->jklbcd', hf.ov, tq2)
            #     - 1 / 12 * einsum('ij,iklabc->jklabc', hf.foo, tt2)
            # )
            # quadruples_r = (
            #     + i_q + 1 / 36 * direct_sum('ai,jklbcd->jklabcd', hf.fov, tt2)
            #     + 1 / 144 * einsum('ba,ijklacde->ijklbcde', hf.fvv, tq2)
            #     - 1 / 144 * einsum('ij, iklmabcd->jklmabcd', hf.foo, tq2)
            # )

            # equation from michael wormit:
            # canonical Hylleraas equation multiplied by factor 4!.
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
            doubles_r = doubles_r.antisymmetrise((0, 1)).antisymmetrise((2, 3))
            if singles_r.select_n_absmax(1)[0][1] > 1e3:
                print("max value of T^S_2 residue too large.")
                print(singles_r.select_n_absmax(3))
                exit()
            if doubles_r.select_n_absmax(1)[0][1] > 1e3:
                print("max value of T^D_2 residue too large.")
                print(doubles_r.select_n_absmax(3))
                exit()

            # update
            ts2 -= 0.5 * singles_r
            td2 -= 1.0 * doubles_r

            # check norm of the residuals
            norm_s = np.sqrt(einsum('ia,ia->', singles_r, singles_r))
            norm_d = np.sqrt(einsum('ijab,ijab->', doubles_r, doubles_r))
            print(f"{i+1}          {norm_s} / {norm_d}")
            if np.sqrt(norm_s ** 2 + norm_d ** 2) < conv_tol:
                print("Converged!")
                break
            elif np.sqrt(norm_s ** 2 + norm_d ** 2) > 1e3:
                print("Second order wavefunction diverged.")
                print(f"Singles norm: {norm_s}. Doubles norm: {norm_d}")
        ts2_can = self.mp2_diffdm.ov
        # sign of the canonical ts2 amplitudes is different!!
        dif = ts2 + ts2_can
        dif_norm = np.sqrt(einsum('ia,ia->', dif, dif))
        max_dif = dif.select_n_absmax(3)
        print("difference to canonical singles amplitudes:")
        print(f"norm: {dif_norm}")
        print(f"max dif: {max_dif}")
        td2_can = self.td2(b.oovv)
        dif = td2 - td2_can
        dif_norm = np.sqrt(einsum('ijab,ijab->', dif, dif))
        max_dif = dif.select_n_absmax(3)
        print("difference to canonical doubles amplitudes:")
        print(f"norm: {dif_norm}")
        print(f"max dif: {max_dif}")
        print("Converged 2nd order single amplitudes:\n", ts2)
        return ts2

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
        # no idea if this gives the correct amplitudes
        """Returns the second order singles amplitudes T^S_2."""
        if space != b.ov:
            raise NotImplementedError("T^S_2 term not implemented ",
                                      f"for space {space}.")
        return self.mp2_diffdm.ov

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
        td = self.t2_with_singles('o1o1v1v1')
        # ret.oo = -0.5 * einsum("ikab,jkab->ij", self.t2oo, self.t2oo)
        ret.oo = -0.5 * einsum("ikab,jkab->ij", td, td)
        ret.ov = -0.5 * (
            + einsum("ijbc,jabc->ia", self.t2oo, hf.ovvv)
            + einsum("jkib,jkab->ia", hf.ooov, self.t2oo)
        ) / self.df(b.ov)
        # ret.vv = 0.5 * einsum("ijac,ijbc->ab", self.t2oo, self.t2oo)
        ret.vv = 0.5 * einsum("ijac,ijbc->ab", td, td)

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
