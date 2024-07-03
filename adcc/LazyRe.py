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
from .functions import direct_sum, einsum
from .GroundState import GroundState
from .misc import cached_member_function
from . import block as b


class LazyRe(GroundState):
    """
    Retaining the excitation degree (RE) ground state class.

    Parameters
    ----------
    hf : ReferenceState
        The SCF reference state.
    conv_tol : float, optional
        Convergence tolerance for the RE ground state amplitudes
        (default: SCF tolerance).
    max_iter : int, optional
        Maximum number of iterations for the iterative determination of the
        RE ground state amplitudes (default: 100).
    """

    def __init__(self, hf, conv_tol=None, max_iter=None):
        if conv_tol is None:
            conv_tol = hf.conv_tol
        self.conv_tol = conv_tol
        if max_iter is None:
            max_iter = 100
        self.max_iter = max_iter
        super().__init__(hf)

    @cached_member_function
    def ts1(self, space):
        """First order RE ground state singles amplitudes.
           Zero for a block diagonal Fock matrix.
        """
        raise NotImplementedError("The first order RE singles amplitudes vanish "
                                  "for a block diagonal fock matrix. Probably you "
                                  "don't need this tensor.")

    @cached_member_function
    def t2(self, space):
        """First order RE ground state doubles amplitudes."""
        from .solver.conjugate_gradient import conjugate_gradient, default_print
        from .solver.preconditioner import JacobiPreconditioner
        from .LazyMp import LazyMp

        if space != b.oovv:
            raise NotImplementedError("First order doubles not implemented for "
                                      f"space {space}.")
        hf = self.reference_state

        # build the right hand side of Ax = b
        rhs = -hf.oovv
        rhs = AmplitudeVector(pphh=rhs)

        # build a guess for the t-amplitudes: use mp-amplitudes as they only
        # scale N^4, while each iteration scales as N^6
        guess = LazyMp(self.reference_state).t2(space)
        guess = AmplitudeVector(pphh=guess)

        print("\nIterating first order RE doubles amplitudes...")
        t2 = conjugate_gradient(Doubles(hf), rhs, guess, callback=default_print,
                                explicit_symmetrisation=None,
                                conv_tol=self.conv_tol,
                                max_iter=self.max_iter,
                                Pinv=JacobiPreconditioner)
        t2 = t2.solution.pphh
        return t2

    @cached_member_function
    def ts2(self, space):
        """Second order RE ground state singles amplitudes."""
        from .solver.conjugate_gradient import conjugate_gradient, default_print
        from .solver.preconditioner import JacobiPreconditioner
        from .LazyMp import LazyMp

        if space != b.ov:
            raise NotImplementedError("Second order singles not implemented for "
                                      f"space {space}.")
        hf = self.reference_state
        t2_1 = self.t2(b.oovv)
        rhs = (
            # N^5: O^2V^3 / N^4: O^1V^3
            - 0.5 * einsum('jabc,ijbc->ia', hf.ovvv, t2_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.5 * einsum('jkib,jkab->ia', hf.ooov, t2_1)
        )
        rhs = AmplitudeVector(ph=rhs)

        # can use MP amplitudes as guess, since the rhs scales N^5 anyway
        guess = LazyMp(self.reference_state).ts2(space)
        guess = AmplitudeVector(ph=guess)

        print("\nIterating Second order RE singles amplitudes...")
        t1 = conjugate_gradient(Singles(hf), rhs, guess, callback=default_print,
                                explicit_symmetrisation=None,
                                conv_tol=self.conv_tol,
                                max_iter=self.max_iter,
                                Pinv=JacobiPreconditioner)
        t1 = t1.solution.ph
        return t1

    @cached_member_function
    def td2(self, space):
        """Second order RE ground state doubles amplitudes.
           Zero as long as the first order singles are 0
           -> Zero for a block diagonal Fock matrix
        """
        raise NotImplementedError("The second order RE doubles amplitudes vanish "
                                  "for a block diagonal fock matrix. Probably you "
                                  "don't need this tensor.")

    @cached_member_function
    def energy_correction(self, level=2):
        """Obtain the RE energy correction at a particular level."""
        hf = self.reference_state
        if level < 2:
            return 0.0
        elif level == 2:
            return -0.25 * hf.oovv.dot(self.t2oo)
        elif level == 3:
            # for a block diagonal fock matrix the third order energy correction
            # vanishes, since the td2 tensor is zero
            return 0.0
        else:
            raise NotImplementedError(f"RE({level}) energy correction "
                                      "not implemented.")

    def energy(self, level=2):
        """
        Obtain the total RE energy (SCF energy plus all corrections)
        at a particular level of perturbation theory.
        """
        # 0th order: SCF; 1st order: zero
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
                               maxlevel=maxlevel, method="RE")

    @property
    def re2_diffdm(self):
        """
        Return the RE2 difference density in the MO basis.
        """
        return self.diffdm(2)

    @property
    def re2_density(self):
        return self.density(2)

    @property
    def re2_dipole_moment(self):
        return self.second_order_dipole_moment


class ReAmplitude(AdcMatrixlike):
    # k fold excited n'th order RE amplitudes are defined according to:
    #   <k|H0|n> - E0 tn_k = - <k|H1|n-1> + sum_{m=1}^{n-1} Em t(n-m)_k
    # The structure of the left hand side is for all orders n the same!
    # Only the right hand side (the inhomogenity) varies with order n.
    def __init__(self, hf):
        self.reference_state = hf

    def __matmul__(self, vec):
        raise NotImplementedError(f"MVP not implemented for {self.__class__}")

    def diagonal(self):
        raise NotImplementedError(f"Diagonal not implemented for {self.__class__}")


class Singles(ReAmplitude):
    def __matmul__(self, vec):
        if isinstance(vec, list):
            return [self.__matmul__(v) for v in vec]
        hf = self.reference_state
        t1 = (einsum('ab,ib->ia', hf.fvv, vec.ph)  # N^3
              - einsum('ij,ja->ia', hf.foo, vec.ph)  # N^3
              - einsum('ibja,jb->ia', hf.ovov, vec.ph))  # N^4
        return AmplitudeVector(ph=t1)

    def diagonal(self):
        hf = self.reference_state
        diag = direct_sum('-i+a->ia', hf.foo.diagonal(), hf.fvv.diagonal())
        return AmplitudeVector(ph=diag.evaluate())


class Doubles(ReAmplitude):
    def __matmul__(self, vec):
        if isinstance(vec, list):
            return [self.__matmul__(v) for v in vec]
        hf = self.reference_state
        t2 = (
            4 * einsum(
                'icka,jkbc->ijab', hf.ovov, vec.pphh
            ).antisymmetrise(0, 1).antisymmetrise(2, 3)
            + 2 * einsum('ac,ijbc->ijab', hf.fvv, vec.pphh).antisymmetrise(2, 3)
            + 2 * einsum('jk,ikab->ijab', hf.foo, vec.pphh).antisymmetrise(0, 1)
            - 0.5 * einsum('abcd,ijcd->ijab', hf.vvvv, vec.pphh)
            - 0.5 * einsum('ijkl,klab->ijab', hf.oooo, vec.pphh)
        )
        return AmplitudeVector(pphh=t2)

    def diagonal(self):
        hf = self.reference_state
        # NOTE: only terms containing the Fock matrix have been considered.
        # For a canonical orbital basis, the diagonal is defined by the
        # usual orbital energy difference.
        diag = direct_sum("+i+j-a-b->ijab",
                          hf.foo.diagonal(), hf.foo.diagonal(),
                          hf.fvv.diagonal(), hf.fvv.diagonal()).symmetrise(2, 3)
        return AmplitudeVector(pphh=diag.evaluate())
