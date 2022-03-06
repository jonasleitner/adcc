from itertools import product
import numpy as np

import libadcc

from .Tensor import Tensor
from .MoSpaces import split_spaces


class ThreeParticleOperator:
    def __init__(self, spaces):
        if hasattr(spaces, "mospaces"):
            self.mospaces = spaces.mospaces
        else:
            self.mospaces = spaces

        if isinstance(spaces, libadcc.ReferenceState):
            self.reference_state = spaces
        elif hasattr(spaces, "reference_state"):
            assert isinstance(spaces.reference_state, libadcc.ReferenceState)
            self.reference_state = spaces.reference_state

        occs = sorted(self.mospaces.subspaces_occupied, reverse=True)
        virts = sorted(self.mospaces.subspaces_virtual, reverse=True)
        self.orbital_subspaces = occs + virts
        assert sum(self.mospaces.n_orbs(ss) for ss in self.orbital_subspaces) \
            == self.mospaces.n_orbs("f")
        combs = list(product(self.orbital_subspaces, repeat=6))
        self.blocks = ["".join(com) for com in combs]
        self._tensors = {}

    @property
    def shape(self):
        """
        Returns the shape tuple of the OneParticleOperator
        """
        size = self.mospaces.n_orbs("f")
        return tuple(size for i in range(6))

    @property
    def size(self):
        """
        Returns the number of elements of the OneParticleOperator
        """
        return np.prod(self.shape)

    @property
    def blocks_nonzero(self):
        """
        Returns a list of the non-zero block labels
        """
        return [b for b in self.blocks if b in self._tensors]

    def is_zero_block(self, block):
        """
        Checks if block is explicitly marked as zero block.
        Returns False if the block does not exist.
        """
        if block not in self.blocks:
            return False
        return block not in self.blocks_nonzero

    def block(self, block):
        """
        Returns tensor of the given block.
        Does not create a block in case it is marked as a zero block.
        Use __getitem__ for that purpose.
        """
        if block not in self.blocks_nonzero:
            raise KeyError("The block function does not support "
                           "access to zero-blocks. Available non-zero "
                           f"blocks are: {self.blocks_nonzero}.")
        return self._tensors[block]

    def __getitem__(self, block):
        if block not in self.blocks:
            raise KeyError(f"Invalid block {block} requested. "
                           f"Available blocks are: {self.blocks}.")
        if block not in self._tensors:
            sym = libadcc.make_symmetry_triples(self.mospaces, block)
            self._tensors[block] = Tensor(sym)
        return self._tensors[block]

    def __getattr__(self, attr):
        from . import block as b
        return self.__getitem__(b.__getattr__(attr))

    def __setitem__(self, block, tensor):
        """
        Assigns a tensor to the specified block
        """
        if block not in self.blocks:
            raise KeyError(f"Invalid block {block} assigned. "
                           f"Available blocks are: {self.blocks}.")
        spaces = split_spaces(block)
        expected_shape = tuple((self.mospaces.n_orbs(s) for s in spaces))
        if expected_shape != tensor.shape:
            raise ValueError("Invalid shape of incoming tensor. "
                             f"Expected shape {expected_shape}, but "
                             f"got shape {tensor.shape} instead.")
        self._tensors[block] = tensor

    def __setattr__(self, attr, value):
        try:
            from . import block as b
            self.__setitem__(b.__getattr__(attr), value)
        except AttributeError:
            super().__setattr__(attr, value)

    def set_zero_block(self, block):
        """
        Set a given block as zero block
        """
        if block not in self.blocks:
            raise KeyError(f"Invalid block {block} set as zero block. "
                           f"Available blocks are: {self.blocks}.")
        self._tensors.pop(block)

    def to_ndarray(self):
        """
        Returns the ThreeParticleOperator as a contiguous
        np.ndarray instance including all blocks
        """
        # offsets to start index of spaces
        offsets = {
            sp: sum(
                self.mospaces.n_orbs(ss)
                for ss in self.orbital_subspaces[:self.orbital_subspaces.index(sp)]
            )
            for sp in self.orbital_subspaces
        }
        # slices for each space
        slices = {
            sp: slice(offsets[sp], offsets[sp] + self.mospaces.n_orbs(sp))
            for sp in self.orbital_subspaces
        }
        ret = np.zeros((self.shape))
        for block in self.blocks_nonzero:
            spaces = split_spaces(block)
            sl = tuple(slices[s] for s in spaces)
            # rowslice, colslice = slices[sp1], slices[sp2]
            dm_block = self[block].to_ndarray()
            # unpacking in subscript -> python3.11
            ret[sl[0], sl[1], sl[2], sl[3], sl[4], sl[5]] = dm_block
        return ret

    def copy(self):
        """
        Return a deep copy of the OneParticleOperator
        """
        ret = ThreeParticleOperator(self.mospaces)
        for b in self.blocks_nonzero:
            ret[b] = self.block(b).copy()
        if hasattr(self, "reference_state"):
            ret.reference_state = self.reference_state
        return ret

    def __iadd__(self, other):
        if self.mospaces != other.mospaces:
            raise ValueError("Cannot add ThreeParticleOperators with "
                             "differing mospaces.")

        for b in other.blocks_nonzero:
            if self.is_zero_block(b):
                self[b] = other.block(b).copy()
            else:
                self[b] = self.block(b) + other.block(b)

        # Update ReferenceState pointer
        if hasattr(self, "reference_state"):
            if hasattr(other, "reference_state") \
                    and self.reference_state != other.reference_state:
                delattr(self, "reference_state")
        return self

    def __add__(self, other):
        return self.copy().__iadd__(other)

    def __isub__(self, other):
        if self.mospaces != other.mospaces:
            raise ValueError("Cannot subtract ThreeParticleOperators with "
                             "differing mospaces.")

        for b in other.blocks_nonzero:
            if self.is_zero_block(b):
                self[b] = -1.0 * other.block(b)  # The copy is implicit
            else:
                self[b] = self.block(b) - other.block(b)

        # Update ReferenceState pointer
        if hasattr(self, "reference_state"):
            if hasattr(other, "reference_state") \
                    and self.reference_state != other.reference_state:
                delattr(self, "reference_state")
        return self

    def __sub__(self, other):
        return self.copy().__isub__(other)

    def evaluate(self):
        for b in self.blocks_nonzero:
            self.block(b).evaluate()
        return self
