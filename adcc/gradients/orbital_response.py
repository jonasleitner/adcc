#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2021 by the adcc authors
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
import adcc.block as b
from adcc.OneParticleOperator import OneParticleOperator
from adcc.functions import direct_sum, einsum, evaluate

from adcc.solver.conjugate_gradient import conjugate_gradient, default_print


class OrbitalResponseMatrix:
    def __init__(self, hf):
        if hf.has_core_occupied_space:
            raise NotImplementedError("OrbitalResponseMatrix not implemented "
                                      "for CVS reference state.")
        self.hf = hf

    @property
    def shape(self):
        no1 = self.hf.n_orbs(b.o)
        nv1 = self.hf.n_orbs(b.v)
        size = no1 * nv1
        return (size, size)

    def __matmul__(self, l_ov):
        ret = (
            + einsum("ab,ib->ia", self.hf.fvv, l_ov)
            - einsum("ij,ja->ia", self.hf.foo, l_ov)
            + einsum("ijab,jb->ia", self.hf.oovv, l_ov)
            - einsum("ibja,jb->ia", self.hf.ovov, l_ov)
        )
        # TODO: generalize once other solvent methods are available
        if "pe_induction_elec" in self.hf.operators.density_dependent_operators:
            # PE contribution to the orbital Hessian
            ops = self.hf.operators
            dm = OneParticleOperator(self.hf, is_symmetric=True)
            dm.ov = l_ov
            ret += ops.density_dependent_operators["pe_induction_elec"](dm).ov
        return evaluate(ret)


class OrbitalResponsePinv:
    def __init__(self, hf):
        if hf.has_core_occupied_space:
            raise NotImplementedError("OrbitalResponsePinv not implemented "
                                      "for CVS reference state.")
        fo = hf.fock(b.oo).diagonal()
        fv = hf.fock(b.vv).diagonal()
        self.df = direct_sum("-i+a->ia", fo, fv).evaluate()

    @property
    def shape(self):
        no1 = self.hf.n_orbs(b.o)
        nv1 = self.hf.n_orbs(b.v)
        size = no1 * nv1
        return (size, size)

    def __matmul__(self, invec):
        return invec / self.df


def orbital_response(hf, rhs):
    """
    Solves the orbital response equations
    for a given reference state and right-hand side
    """
    A = OrbitalResponseMatrix(hf)
    Pinv = OrbitalResponsePinv(hf)
    x0 = (Pinv @ rhs).evaluate()
    l_ov = conjugate_gradient(A, rhs=rhs, x0=x0, Pinv=Pinv,
                              explicit_symmetrisation=None,
                              callback=default_print)
    return l_ov.solution
