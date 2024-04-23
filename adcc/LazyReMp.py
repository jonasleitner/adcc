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
from .AdcMatrix import AdcMatrixlike
from .AmplitudeVector import AmplitudeVector
from .functions import einsum, direct_sum
from .GroundState import GroundState
from .misc import cached_member_function
from . import block as b


class LazyReMp(GroundState):
    def __init__(self, hf, remp_A, conv_tol=None, max_iter=None):
        """Initialise the REMP ground state class."""
        if conv_tol is None:
            conv_tol = hf.conv_tol
        self.conv_tol = conv_tol
        if max_iter is None:
            max_iter = 100
        self.max_iter = max_iter
        # TODO: default value for remp_A?
        self.remp_A = remp_A
        super().__init__(hf)

    @cached_member_function
    def ts1(self, space):
        """First order REMP ground state singles amplitudes.
           Zero for a block diagonal Fock matrix.
        """
        raise RuntimeError("The first order REMP singles amplitudes vanish for a "
                           "block diagonal fock matrix. Probably you don't need "
                           "this tensor.")
        from .solver.conjugate_gradient import conjugate_gradient, default_print
        from .solver.preconditioner import JacobiPreconditioner

        if space != b.ov:
            raise NotImplementedError("First order singles not implemented for "
                                      f"space {space}.")
        hf = self.reference_state
        # build the right hand side of Ax = b
        rhs = - (1 - self.remp_A) * hf.fov
        rhs = AmplitudeVector(ph=rhs)

        # don't use zero guess -> division by 0 error
        guess = hf.fov.ones_like() * 1e-6
        guess = AmplitudeVector(ph=guess)

        print("\nIterating first order REMP singles amplitudes...")
        t1 = conjugate_gradient(Singles(hf, self.remp_A), rhs, guess,
                                callback=default_print,
                                explicit_symmetrisation=None,
                                conv_tol=self.conv_tol, max_iter=self.max_iter,
                                Pinv=JacobiPreconditioner)
        t1 = t1.solution.ph
        return t1

    @cached_member_function
    def t2(self, space):
        """First order REMP ground state doubles amplitudes."""
        from .solver.conjugate_gradient import conjugate_gradient, default_print
        from .solver.preconditioner import JacobiPreconditioner
        from .LazyMp import LazyMp

        if space != b.oovv:
            raise NotImplementedError("First order doubles not implemented for "
                                      f"space {space}.")
        hf = self.reference_state
        # build the right hand side
        rhs = - hf.oovv
        rhs = AmplitudeVector(pphh=rhs)
        # guess: use MP t-amplitude (only single N^4 contraction)
        guess = LazyMp(self.reference_state).t2(space)
        guess = AmplitudeVector(pphh=guess)

        print("\nIterating first order REMP doubles amplitudes...")
        t2 = conjugate_gradient(Doubles(hf, self.remp_A), rhs, guess,
                                callback=default_print,
                                explicit_symmetrisation=None,
                                conv_tol=self.conv_tol,
                                max_iter=self.max_iter,
                                Pinv=JacobiPreconditioner)
        t2 = t2.solution.pphh
        return t2

    @cached_member_function
    def energy_correction(self, level=2):
        """Obtain the REMP energy correction at a particular level."""
        hf = self.reference_state
        if level < 2:
            return 0.0
        elif level == 2:
            return -0.25 * hf.oovv.dot(self.t2oo)
        else:
            raise NotImplementedError(f"REMP({level}) energy correction not "
                                      "implemented.")

    def energy(self, level=2):
        """
        Obtian the total REMP energy (SCF energy plus all corrections)
        at a particular level of perturbation theory.
        """
        if level == 0:
            # f_ii - 0.5 * (1 - A) * <ij||ij>
            raise NotImplementedError("Total REMP(0) energy not implemented.")
        # first order contribution = SCF energy
        energies = [self.reference_state.energy_scf]
        for order in range(2, level + 1):
            energies.append(self.energy_correction(order))
        return sum(energies)

    def to_qcvars(self, properties=False, recurse=False, maxlevel=2):
        """
        Return a dictionary with property keys compatible to a Psi4 wavefunction
        or a QCEngine Atomicresults object.
        """
        return self._to_qcvars(properties=properties, recurse=recurse,
                               maxlevel=maxlevel, method="REMP")

    @property
    def remp2_diffdm(self):
        """Return the REMP2 difference density in the MO basis."""
        return self.diffdm(2)

    @property
    def remp2_density(self):
        return self.density(2)

    @property
    def remp2_dipole_moment(self):
        return self.second_order_dipole_moment


class ReMpAmplitude(AdcMatrixlike):
    def __init__(self, hf, remp_A) -> None:
        self.reference_state = hf
        self.remp_A = remp_A

    def __matmul__(self, vec):
        raise NotImplementedError(f"MVP not implemented for {self.__class__}")

    def diagonal(self):
        raise NotImplementedError(f"Diagonal not implemented for {self.__class__}")


class Singles(ReMpAmplitude):
    def __matmul__(self, vec):
        if isinstance(vec, list):
            return [self.__matmul__(v) for v in vec]
        hf = self.reference_state
        t1 = (einsum('ab,ib->ia', hf.fvv, vec.ph)
              - einsum('ij,ja->ia', hf.foo, vec.ph)
              # plus an additional term from the MP residual containing first order
              # doubles: - A * f_jb * t_ij^ab
              - (1 - self.remp_A) * einsum('ibja,jb->ia', hf.ovov, vec.ph))
        return AmplitudeVector(ph=t1)

    def diagonal(self):
        hf = self.reference_state
        diag = direct_sum('-i+a->ia', hf.foo.diagonal(), hf.fvv.diagonal())
        return AmplitudeVector(ph=diag.evaluate())


class Doubles(ReMpAmplitude):
    def __matmul__(self, vec):
        if isinstance(vec, list):
            return [self.__matmul__(v) for v in vec]
        hf = self.reference_state
        t2 = (
            + 2 * einsum('ac,ijbc->ijab', hf.fvv, vec.pphh).antisymmetrise(2, 3)
            + 2 * einsum('jk,ikab->ijab', hf.foo, vec.pphh).antisymmetrise(0, 1)
            + (1 - self.remp_A) * (
                + 4 * einsum(
                    'icka,jkbc->ijab', hf.ovov, vec.pphh  # N^6
                ).antisymmetrise(0, 1).antisymmetrise(2, 3)
                - 0.5 * einsum('abcd,ijcd->ijab', hf.vvvv, vec.pphh)  # N^6
                - 0.5 * einsum('ijkl,klab->ijab', hf.oooo, vec.pphh)  # N^6
            )
        )
        return AmplitudeVector(pphh=t2)

    def diagonal(self):
        hf = self.reference_state
        occ = hf.foo.diagonal()
        virt = hf.fvv.diagonal()
        diag = direct_sum('+i+j-a-b->ijab', occ, occ, virt, virt).symmetrise(2, 3)
        return AmplitudeVector(pphh=diag.evaluate())
