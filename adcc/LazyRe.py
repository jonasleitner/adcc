from .LazyMp import LazyMp
from . import block as b
from .functions import einsum, direct_sum
from .misc import cached_member_function
from .AmplitudeVector import AmplitudeVector
from .AdcMatrix import AdcMatrixlike


class LazyRe(LazyMp):
    @cached_member_function
    def t1(self, space):
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

        # guess: 0 will be instantly converged for canonical orbtitals
        #   -> use 1 guess
        guess = hf.fov.ones_like()
        guess = AmplitudeVector(ph=guess)

        t1 = conjugate_gradient(t1_1(hf), rhs, guess, callback=default_print,
                                explicit_symmetrisation=None,
                                Pinv=JacobiPreconditioner)
        t1 = t1.solution.ph
        return t1

    @cached_member_function
    def t2(self, space):
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

        # build a guess for the t-amplitudes: use mp-amplitudes for now
        guess = super().t2(space)
        guess = AmplitudeVector(pphh=guess)

        print("\nIterating RE T2 amplitudes")
        t2 = conjugate_gradient(t2_1(hf), rhs, guess, callback=default_print,
                                explicit_symmetrisation=None, conv_tol=1e-12,
                                Pinv=JacobiPreconditioner)
        t2 = t2.solution.pphh
        return t2


class doubles_sym:
    def symmetrise(self, vec):
        if isinstance(vec, list):
            return [self.symmetrise(v) for v in vec]
        vec['pphh'] = vec.pphh.antisymmetrise(0, 1).antisymmetrise(2, 3)
        return vec


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
