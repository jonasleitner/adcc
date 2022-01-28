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


class helper_dict(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(self):
        return self


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
        """iterative T2 amplitudes through minimization
           of the Hylleraas functional
           """
        print(f"\nComputing iterative T2 amplitudes for space {space}")
        hf = self.reference_state
        sp = split_spaces(space)
        assert all(s == b.v for s in sp[2:])
        # eia = self.df(sp[0] + b.v)
        # ejb = self.df(sp[1] + b.v)
        # try complete sym for guess
        # delta = direct_sum("ia+jb->ijab", eia, ejb)
        # delta = delta.symmetrise((0, 1)).symmetrise((2, 3))

        # guess setup
        t2_amp = hf.eri(space)

        maxiter = 100
        conv_tol = 1e-15
        print("iteration, residue norm D/S")
        for i in range(maxiter):
            # sum_c(t_ijac f_cb - t_ijbc f_ca) = 2 * sum_c t_ijac f_cb
            # - sum_k(t_ikab f_jk - t_jkab f_ik) = - 2 * sum_k t_ikab f_jk
            residue = 2.0 * \
                einsum("ijac,cb->ijab", t2_amp, hf.fvv) \
                - 2.0 * \
                einsum("ikab,jk->ijab", t2_amp, hf.foo) - \
                hf.eri(space)
            residue = residue.antisymmetrise((0, 1)).antisymmetrise((2, 3))

            if residue.select_n_absmax(1)[0][1] > 1e3:
                print("max value of residue to large")
                print(residue.select_n_absmax(3))
                exit()
            # add residue to t2_amplutides
            t2_amp = t2_amp - 0.25 * residue

            # E = -0.5 * einsum('ijab,ijab->', t2_amp, hf.oovv) \
            #    -0.5 * einsum('ikab,ijab,jk->', t2_amp, t2_amp, hf.foo) \
            #    +0.5 * einsum('ijac,ijab,cb->', t2_amp, t2_amp, hf.fvv)
            # print("current correction: ", E)

            # compute the norm of the residue
            norm = np.sqrt(einsum("ijab,ijab->", residue, residue))
            print(f"{i+1}         {norm}")
            if norm < conv_tol:
                print("Converged!")
                break
            elif norm > 1e3:
                print("diverged :(")
                exit()

        # compare to canonical t2_amplitudes:
        diff = t2_amp - self.t2(space)
        diff_norm = np.sqrt(einsum('ijab,ijab->', diff, diff))
        print("diff Hyl-RSPT amps: norm = ", diff_norm)
        print("diff Hyl-RSPT amps: max val = ", diff.select_n_absmax(3))
        # if diff_norm > 1e-13:
        #    print(diff.to_ndarray())
        return t2_amp

    @cached_member_function
    def t2_with_singles(self, space):
        """iterative first order T amplitudes (including singles) through
           minimization of the Hylleraas functional
           """

        print("\nComputing iterative T amplitudes (including singles) for space",
              space)
        hf = self.reference_state
        sp = split_spaces(space)
        assert all(s == b.v for s in sp[2:])
        eia = self.df(sp[0] + b.v)

        # guess setup
        td_amp = hf.eri(space)
        ts_amp = eia

        maxiter = 100
        conv_tol = 1e-15
        print("iteration, residue norm Doubles/Singles")
        for i in range(maxiter):
            # total residue (with singles):
            doubles_r = einsum('ia,jb->ijab', ts_amp, hf.fov) - \
                0.25 * hf.eri(space) - \
                0.5 * einsum('ikab,jk->ijab', td_amp, hf.foo) + \
                0.5 * einsum('ijac,cb->ijab', td_amp, hf.fvv)
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
            td_amp = td_amp - 1.0 * doubles_r
            ts_amp -= 0.25 * singles_r

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
        # if diff_norm > 1e-13:
        #     print(diff.to_ndarray())
        print("converged singles amplitudes:\n", ts_amp)
        return helper_dict(singles=ts_amp, doubles=td_amp)

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
        ret.oo = -0.5 * einsum("ikab,jkab->ij", self.t2oo, self.t2oo)
        ret.ov = -0.5 * (
            + einsum("ijbc,jabc->ia", self.t2oo, hf.ovvv)
            + einsum("jkib,jkab->ia", hf.ooov, self.t2oo)
        ) / self.df(b.ov)
        ret.vv = 0.5 * einsum("ijac,ijbc->ab", self.t2oo, self.t2oo)

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
            amps = self.t2_with_singles('o1o1v1v1')
            ts = amps["singles"]
            td = amps["doubles"]
            i1 = - einsum('ja,ia,ij->', ts, ts, hf.foo)
            i2 = + einsum('ib,ia,ba->', ts, ts, hf.fvv)
            i3 = + einsum('ia,ijab,jb->', ts, td, hf.fov)
            i4 = + einsum('ijab,ia,bj->', td, ts, hf.fvo)
            i5 = - 0.5 * einsum('ijab,ijab->', td, hf.oovv)
            i6 = - 0.5 * einsum('ikab,ijab,jk->', td, td, hf.foo)
            i7 = + 0.5 * einsum('ijac,ijab,cb->', td, td, hf.fvv)
            print(i1, i2, i3, i4, i5, i6, i7)
            print("can merge i3 and i4?: ", i3 - i4)
            return i1 + i2 + i3 + i4 + i5 + i6 + i7

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
