from .LazyMp import LazyMp
from . import block as b
from .functions import einsum, direct_sum
from .misc import cached_member_function
from .AmplitudeVector import AmplitudeVector
from .AdcMatrix import AdcMatrixlike


class LazyRe(LazyMp):
    def __init__(self, hf, re_conv_tol=None, re_max_iter=None):
        """Initialise the retaining the excitation degree (RE) ground state class.
        """
        if re_conv_tol is None:
            re_conv_tol = hf.conv_tol
        self.re_conv_tol = re_conv_tol
        if re_max_iter is None:
            re_max_iter = 100
        self.re_max_iter = re_max_iter
        super().__init__(hf)

    @cached_member_function
    def ts1(self, space):
        """First order RE ground state singles amplitudes.
           Zero for a block diagonal Fock matrix.
        """
        raise RuntimeError("The first order RE singles amplitudes vanish for a "
                           "block diagonal fock matrix. Probably you don't need "
                           "this tensor.")
        # can't import on top -> circular import
        from .solver.conjugate_gradient import conjugate_gradient, default_print
        from .solver.preconditioner import JacobiPreconditioner

        if space != b.ov:
            raise NotImplementedError("First order singles not implemented for "
                                      f"space {space}.")
        hf = self.reference_state

        # build the right hand side of Ax = b
        rhs = -hf.fov
        rhs = AmplitudeVector(ph=rhs)

        # can't use zero guess: divide by 0 in conjugate gradient for canonical
        # orbital basis -> i think random is better than one guess
        guess = hf.fov.zeros_like()
        guess = AmplitudeVector(ph=guess.set_random())

        print("\nIterating first order RE singles amplitudes...")
        t1 = conjugate_gradient(Singles(hf), rhs, guess, callback=default_print,
                                explicit_symmetrisation=None,
                                conv_tol=self.re_conv_tol,
                                max_iter=self.re_max_iter,
                                Pinv=JacobiPreconditioner)
        t1 = t1.solution.ph
        return t1

    @cached_member_function
    def t2(self, space):
        """First order RE ground state doubles amplitudes."""
        # can't import on top -> circular import
        from .solver.conjugate_gradient import conjugate_gradient, default_print
        from .solver.preconditioner import JacobiPreconditioner

        if space != b.oovv:
            raise NotImplementedError("First order doubles not implemented for "
                                      f"space {space}.")
        hf = self.reference_state

        # build the right hand side of Ax = b
        rhs = -hf.oovv
        rhs = AmplitudeVector(pphh=rhs)

        # build a guess for the t-amplitudes: use mp-amplitudes as they only
        # scale N^4, while each iteration scales as N^6
        guess = super().t2(space)
        guess = AmplitudeVector(pphh=guess)

        print("\nIterating first order RE doubles amplitudes...")
        t2 = conjugate_gradient(Doubles(hf), rhs, guess, callback=default_print,
                                explicit_symmetrisation=None,
                                conv_tol=self.re_conv_tol,
                                max_iter=self.re_max_iter,
                                Pinv=JacobiPreconditioner)
        t2 = t2.solution.pphh
        return t2

    @cached_member_function
    def ts2(self, space):
        """Second order RE ground state singles amplitudes"""
        # can't import on top -> circular import
        from .solver.conjugate_gradient import conjugate_gradient, default_print
        from .solver.preconditioner import JacobiPreconditioner

        if space != b.ov:
            raise NotImplementedError("Second order singles not implemented for "
                                      f"space {space}.")
        hf = self.reference_state
        t2_1 = self.t2(b.oovv)
        rhs = (
            # N^5: O^2V^3 / N^4: O^1V^3
            + 0.5 * einsum('jabc,ijbc->ia', hf.ovvv, t2_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum('jkib,jkab->ia', hf.ooov, t2_1)
            - 1 * einsum('jb,ijab->ia', hf.fov, t2_1)  # N^4: O^2V^2 / N^4: O^2V^2
        )
        rhs = AmplitudeVector(ph=rhs)

        # zero guess leads to division by zero for small systems
        guess = hf.fov.ones_like() * 1e-6
        guess = AmplitudeVector(ph=guess)

        print("\nIterating Second order RE singles amplitudes...")
        t1 = conjugate_gradient(Singles(hf), rhs, guess, callback=default_print,
                                explicit_symmetrisation=None,
                                conv_tol=self.re_conv_tol,
                                max_iter=self.re_max_iter,
                                Pinv=JacobiPreconditioner)
        t1 = t1.solution.ph
        return t1

    @cached_member_function
    def td2(self, space):
        """Second order RE ground state doubles amplitudes
           Zero as long as the first order singles are 0
           -> Zero for a block diagonal Fock matrix
        """
        raise RuntimeError("The second order RE doubles amplitudes vanish for a "
                           "block diagonal fock matrix. Probably you don't need "
                           "this tensor.")
        # can't import on top -> circular import
        from .solver.conjugate_gradient import conjugate_gradient, default_print
        from .solver.preconditioner import JacobiPreconditioner

        if space != b.oovv:
            raise NotImplementedError("First order doubles not implemented for "
                                      f"space {space}.")
        hf = self.reference_state
        # build the right hand side of Ax = b
        rhs = hf.oovv.zeros_like()
        rhs = AmplitudeVector(pphh=rhs)
        # build a guess for the t-amplitudes: use mp-amplitudes as they
        # scale N^6 just as the iteration scheme
        guess = super().td2(space)
        guess = AmplitudeVector(pphh=guess)

        print("\nIterating Second order RE doubles amplitudes...")
        t2 = conjugate_gradient(Doubles(hf), rhs, guess, callback=default_print,
                                explicit_symmetrisation=None,
                                conv_tol=self.re_conv_tol,
                                max_iter=self.re_max_iter,
                                Pinv=JacobiPreconditioner)
        t2 = t2.solution.pphh
        return t2

    @property
    def re2_diffdm(self):
        return super().mp2_diffdm

    @cached_member_function
    def energy_correction(self, level=2):
        """Obtain the RE energy correction at a particular level"""
        if level > 3:
            raise NotImplementedError(f"RE({level}) energy correction "
                                      "not implemented.")
        if level < 2:
            return 0.0
        hf = self.reference_state
        if level == 2:
            return -0.25 * hf.oovv.dot(self.t2oo)
        elif level == 3:
            # for a block diagonal fock matrix the third order energy correction
            # vanishes, since the td2 tensor is zero
            return 0.0

    @cached_member_function
    def energy(self, level=2):
        """Obtain the total energy (SCF energy plus all corrections)
           at a particular level of perturbation theory.
        """
        # We already have in 0th order the HF energy for RE
        energies = [self.reference_state.energy_scf]
        for order in range(2, level + 1):
            energies.append(self.energy_correction(order))
        return sum(energies)

    @property
    def re2_density(self):
        return self.density(2)

    @property
    def re2_dipole_moment(self):
        return super().mp2_dipole_moment


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
        t1 = (einsum('ab,ib->ia', hf.fvv, vec.ph)
              - einsum('ij,ja->ia', hf.foo, vec.ph)
              - einsum('ibja,jb->ia', hf.ovov, vec.ph))
        return AmplitudeVector(ph=t1)

    def diagonal(self):
        hf = self.reference_state
        diag = direct_sum('+i-a->ia', hf.foo.diagonal(), hf.fvv.diagonal())
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
