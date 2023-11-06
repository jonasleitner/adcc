from .LazyMp import LazyMp
from . import block as b
from .functions import einsum
from .misc import cached_member_function
from .solver.conjugate_gradient import conjugate_gradient, default_print
from .AmplitudeVector import AmplitudeVector


class LazyRe(LazyMp):
    @cached_member_function
    def t1(self, space):
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
                                explicit_symmetrisation=None)
        t1 = t1.solution.ph
        print("norm: ", einsum('ia,ia->', t1, t1))
        return t1

    @cached_member_function
    def t2(self, space):
        if space != b.oovv:
            raise NotImplementedError("First order doubles not implemented for "
                                      f"space {space}.")
        hf = self.reference_state

        # build the right hand side of Ax = b
        rhs = -hf.oovv
        rhs = AmplitudeVector(pphh=rhs)

        # build a guess for the t-amplitudes: use mp-amplitudes for now
        guess = super(LazyRe, self).t2(space)
        guess = AmplitudeVector(pphh=guess)

        # TODO: what about explicit symmetrisation?
        t2 = conjugate_gradient(t2_1(hf), rhs, guess, callback=default_print,
                                explicit_symmetrisation=None)
        t2 = t2.solution.pphh
        print("norm: ", einsum('ijab,ijab->', t2, t2))
        return t2


class doubles_sym:
    def symmetrise(self, vec):
        if isinstance(vec, list):
            return [self.symmetrise(v) for v in vec]
        vec['pphh'] = vec.pphh.antisymmetrise(0, 1).antisymmetrise(2, 3)
        return vec


class ReAmplitude:
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


class t2_1(ReAmplitude):
    def __matmul__(self, vec):
        if isinstance(vec, list):
            return [self.__matmul__(v) for v in vec]
        hf = self.reference_state
        t2 = (
            4 * einsum('icka,jkbc->ijab', hf.ovov, vec.pphh).antisymmetrise(0, 1).antisymmetrise(2, 3)  # noqa E501
            + 2 * einsum('ac,ijbc->ijab', hf.fvv, vec.pphh).antisymmetrise(2, 3)
            + 2 * einsum('jk,ikab->ijab', hf.foo, vec.pphh).antisymmetrise(0, 1)
            - 0.5 * einsum('abcd,ijcd->ijab', hf.vvvv, vec.pphh)
            - 0.5 * einsum('ijkl,klab->ijab', hf.oooo, vec.pphh)
        )
        return AmplitudeVector(pphh=t2)
