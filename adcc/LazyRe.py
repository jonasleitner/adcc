from .LazyMp import LazyMp
from . import block as b
from .functions import einsum, direct_sum
from .misc import cached_member_function
from .AmplitudeVector import AmplitudeVector
from .AdcMatrix import AdcMatrixlike


class LazyRe(LazyMp):
    # As long as the first order singles are neglected the energy and
    # density matrix expressions are the same (only checked through second order)
    # as for the MP ground state.
    # Therefore, only the methods for the determination of t-amplitudes
    # are defined on the LazyRe class.

    def __init__(self, hf, remp_conv_tol=None):
        """Initialise the retaining the excitation degree (RE) ground state class.
        """
        # TODO: What is a good default value for the convergence tolerance?
        # excitation energies seem to be insensitive to the convergence tolerance
        # as even a tolerance of 1e-5 only introduces a error of 1e-6eV for
        # H2O cc-pvtz.
        if remp_conv_tol is None:
            remp_conv_tol = hf.conv_tol
        self.remp_conv_tol = remp_conv_tol
        super().__init__(hf)

    @cached_member_function
    def t1(self, space):
        """First order RE ground state singles amplitudes."""
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

        # zero guess leads to instant convergence for canonical orbitals
        guess = hf.fov.zeros_like()
        guess = AmplitudeVector(ph=guess)

        print("\nIterating first order RE singles amplitudes...")
        t1 = conjugate_gradient(t1_1(hf), rhs, guess, callback=default_print,
                                conv_tol=self.remp_conv_tol,
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
        t2 = conjugate_gradient(t2_1(hf), rhs, guess, callback=default_print,
                                conv_tol=self.remp_conv_tol,
                                Pinv=JacobiPreconditioner)
        t2 = t2.solution.pphh
        return t2

    def td2(self, space):
        """Second order RE ground state doubles amplitudes"""
        raise NotImplementedError("Second order RE Doubles amplitudes not "
                                  "implemented")


class ReAmplitude(AdcMatrixlike):
    def __init__(self, hf):
        self.reference_state = hf

    def __matmul__(self, vec):
        raise NotImplementedError(f"MVP not implemented for {self}")


class t1_1(ReAmplitude):
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


class t2_1(ReAmplitude):
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
        # NOTE: only terms containing the Fock matrix have been considered
        # for a canonical orbital basis, the diagonal is defined by the
        # usual orbital energy difference.
        diag = direct_sum("+i+j-a-b->ijab",
                          hf.foo.diagonal(), hf.foo.diagonal(),
                          hf.fvv.diagonal(), hf.fvv.diagonal()).symmetrise(2, 3)
        return AmplitudeVector(pphh=diag.evaluate())
