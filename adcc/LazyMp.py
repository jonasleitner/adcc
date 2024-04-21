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
from .functions import direct_sum, einsum
from .GroundState import GroundState
from .misc import cached_member_function
from .OneParticleOperator import OneParticleOperator
from .Intermediates import register_as_intermediate
from .MoSpaces import split_spaces
from . import block as b


class LazyMp(GroundState):
    @cached_member_function
    def t2(self, space):
        """T2 amplitudes"""
        hf = self.reference_state
        sp = split_spaces(space)
        assert all(s == b.v for s in sp[2:])
        eia = self.df(sp[0] + b.v)
        ejb = self.df(sp[1] + b.v)
        return (
            hf.eri(space) / direct_sum("ia+jb->ijab", eia, ejb).symmetrise((2, 3))
        )

    @cached_member_function
    def ts2(self, space):
        """Computes the second order singles amplitudes."""
        if space != b.ov:
            raise NotImplementedError("Second order singles amplitudes not "
                                      f"implemented for space {space}.")
        hf = self.reference_state
        return -0.5 * (
            + einsum("ijbc,jabc->ia", self.t2oo, hf.ovvv)
            + einsum("jkib,jkab->ia", hf.ooov, self.t2oo)
        ) / self.df(b.ov)

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
        Obtain the total MP energy (SCF energy plus all corrections)
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

    def to_qcvars(self, properties=False, recurse=False, maxlevel=2):
        """
        Return a dictionary with property keys compatible to a Psi4 wavefunction
        or a QCEngine Atomicresults object.
        """
        return self._to_qcvars(properties=properties, recurse=recurse,
                               maxlevel=maxlevel, method="MP")

    @property
    def mp2_diffdm(self):
        """
        Return the MP2 difference density in the MO basis.
        """
        return self.diffdm(2)

    @property
    def mp2_density(self):
        return self.density(2)

    @property
    def mp2_dipole_moment(self):
        return self.second_order_dipole_moment


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
