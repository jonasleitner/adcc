#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
import numpy as np

import libadcc

from .LazyMp import LazyMp
from .AdcMethod import AdcMethod
from .functions import empty_like
from .AmplitudeVector import AmplitudeVector


class AdcMatrix(libadcc.AdcMatrix):
    def __init__(self, method, mp_results):
        """
        Initialise an ADC matrix from a method, the reference_state
        and appropriate MP results.
        """
        if not isinstance(method, AdcMethod):
            method = AdcMethod(method)
        if isinstance(mp_results, (libadcc.ReferenceState,
                                   libadcc.HartreeFockSolution_i)):
            mp_results = LazyMp(mp_results)
        if not isinstance(mp_results, libadcc.LazyMp):
            raise TypeError("mp_results is not a valid object. It needs to be "
                            "either a LazyMp, a ReferenceState or a "
                            "HartreeFockSolution_i.")

        self.method = method
        super().__init__(method.name, mp_results)

    def compute_matvec(self, in_ampl, out_ampl=None):
        """
        Compute the matrix-vector product of the ADC matrix
        with an excitation amplitude and return the result
        in the out_ampl if it is given, else the result
        will be returned.
        """
        if out_ampl is None:
            out_ampl = empty_like(in_ampl)
        elif not isinstance(out_ampl, type(in_ampl)):
            raise TypeError("Types of in_ample and out_ampl do not match.")
        if not isinstance(in_ampl, AmplitudeVector):
            raise TypeError("in_ampl has to be of type AmplitudeVector.")
        else:
            super().compute_matvec(in_ampl.to_cpp(), out_ampl.to_cpp())
        return out_ampl

    @property
    def ndim(self):
        return 2

    def matvec(self, v):
        out = empty_like(v)
        self.compute_matvec(v, out)
        return out

    def rmatvec(self, v):
        # ADC matrix is symmetric
        return self.matvec(self, v)

    def __matmul__(self, other):
        if isinstance(other, AmplitudeVector):
            return self.compute_matvec(other)
        if isinstance(other, list):
            if all(isinstance(elem, AmplitudeVector) for elem in other):
                return [self.compute_matvec(ov) for ov in other]
        return NotImplemented

    def __repr__(self):
        return "AdcMatrix(method={})".format(self.method.name)

    def construct_symmetrisation_for_blocks(self):
        """
        Construct the symmetrisation functions, which need to be
        applied to relevant blocks of an AmplitudeVector in order
        to symmetrise it to the right symmetry in order to be used
        with the various matrix-vector-products of this function.

        Most importantly the returned functions antisymmetrise
        the occupied and virtual parts of the doubles parts
        if this is sensible for the method behind this adcmatrix.

        Returns a dictionary block identifier -> function
        """
        ret = {}
        if self.method.is_core_valence_separated:
            def symmetrise_cvs_adc_doubles(invec, outvec):
                # CVS doubles part is antisymmetric wrt. (i,K,a,b) <-> (i,K,b,a)
                invec.antisymmetrise_to(outvec, [(2, 3)])
            ret["d"] = symmetrise_cvs_adc_doubles
        else:
            def symmetrise_generic_adc_doubles(invec, outvec):
                scratch = empty_like(outvec)
                # doubles part is antisymmetric wrt. (i,j,a,b) <-> (i,j,b,a)
                invec.antisymmetrise_to(scratch, [(2, 3)])
                # doubles part is symmetric wrt. (i,j,a,b) <-> (j,i,b,a)
                scratch.symmetrise_to(outvec, [(0, 1), (2, 3)])
            ret["d"] = symmetrise_generic_adc_doubles
        return ret

    def dense_basis(self, blocks=None):
        """
        Return the list of indices and their values
        of the dense basis representation
        """
        ret = []
        if blocks is None:
            blocks = self.blocks

        if "s" in blocks:
            sp_s = self.block_spaces("s")
            n_orbs_s = [self.mospaces.n_orbs(sp) for sp in sp_s]
            for i in range(n_orbs_s[0]):
                for a in range(n_orbs_s[1]):
                    ret.append([((i, a), 1)])

        if "d" in blocks:
            sp_d = self.block_spaces("d")

            if sp_d[0] == sp_d[1] and sp_d[2] == sp_d[3]:
                nso = self.mospaces.n_orbs(sp_d[0])
                nsv = self.mospaces.n_orbs(sp_d[2])
                ret.extend([[((i, j, a, b), +1 / 2),
                             ((j, i, a, b), -1 / 2),
                             ((i, j, b, a), -1 / 2),
                             ((j, i, b, a), +1 / 2)]
                            for i in range(nso) for j in range(i)
                            for a in range(nsv) for b in range(a)])
            elif sp_d[2] == sp_d[3]:
                nso = self.mospaces.n_orbs(sp_d[0])
                nsc = self.mospaces.n_orbs(sp_d[1])
                nsv = self.mospaces.n_orbs(sp_d[2])
                ret.extend([[((i, j, a, b), +1 / np.sqrt(2)),
                             ((i, j, b, a), -1 / np.sqrt(2))]
                            for i in range(nso) for j in range(nsc)
                            for a in range(nsv) for b in range(a)])
            else:
                nso = self.mospaces.n_orbs(sp_d[0])
                nsc = self.mospaces.n_orbs(sp_d[1])
                nsv = self.mospaces.n_orbs(sp_d[2])
                nsw = self.mospaces.n_orbs(sp_d[3])
                ret.append([((i, j, b, a), 1)
                            for i in range(nso) for j in range(nsc)
                            for a in range(nsv) for b in range(nsw)])

        if any(b not in "sd" for b in self.blocks):
            raise NotImplementedError("Blocks other than s and d "
                                      "not implemented")
        return ret

    def to_dense_matrix(self, out=None):
        """
        Return the ADC matrix object as a dense numpy array. Converts the sparse
        internal representation of the ADC matrix to a dense matrix and return
        as a numpy array.

        Notes
        -----

        This method is only intended to be used for debugging and
        visualisation purposes as it involves computing a large amount of
        matrix-vector products and the returned array consumes a considerable
        amount of memory.

        The resulting matrix has no spin symmetry imposed, which means that
        its eigenspectrum may contain non-physical excitations (e.g. with linear
        combinations of α->β and α->α components in the excitation vector).

        This function has not been sufficiently tested to be considered stable.
        """
        from adcc import guess_zero

        import tqdm

        # Get zero amplitude of the appropriate symmetry
        # (TODO: Only true for C1, where there is only a single irrep)
        ampl_zero = guess_zero(self)
        assert self.mospaces.point_group == "C1"

        # Build the shape of the returned array
        # Since the basis of the doubles block is not the unit vectors
        # this *not* equal to the shape of the AdcMatrix object
        basis = {b: self.dense_basis(b) for b in self.blocks}
        mat_len = sum(len(basis[b]) for b in basis)

        if out is None:
            out = np.zeros((mat_len, mat_len))
        else:
            if out.shape != (mat_len, mat_len):
                raise ValueError("Output array has shape ({0:}, {1:}), but "
                                 "shape ({2:}, {2:}) is required."
                                 "".format(*out.shape, mat_len))
            out[:] = 0  # Zero all data in out.

        # Check for the cases actually implemented
        if any(b not in "sd" for b in self.blocks):
            raise NotImplementedError("Blocks other than s and d "
                                      "not implemented")
        if "s" not in self.blocks:
            raise NotImplementedError("Block 's' needs to be present")

        # Extract singles-singles block (contiguous)
        assert "s" in self.blocks
        n_orbs_s = [self.mospaces.n_orbs(sp) for sp in self.block_spaces("s")]
        n_s = np.prod(n_orbs_s)
        assert len(basis["s"]) == n_s
        view_ss = out[:n_s, :n_s].reshape(*n_orbs_s, *n_orbs_s)
        for i in range(n_orbs_s[0]):
            for a in range(n_orbs_s[1]):
                ampl = ampl_zero.copy()
                ampl["s"][i, a] = 1
                view_ss[:, :, i, a] = (self @ ampl)["s"].to_ndarray()

        # Extract singles-doubles and doubles-doubles block
        if "d" in self.blocks:
            assert self.blocks == ["s", "d"]
            view_sd = out[:n_s, n_s:].reshape(*n_orbs_s, len(basis["d"]))
            view_dd = out[n_s:, n_s:]
            for j, bas1 in tqdm.tqdm(enumerate(basis["d"]),
                                     total=len(basis["d"])):
                ampl = ampl_zero.copy()
                for idx, val in bas1:
                    ampl["d"][idx] = val
                ret_ampl = self @ ampl
                view_sd[:, :, j] = ret_ampl["s"].to_ndarray()

                for i, bas2 in enumerate(basis["d"]):
                    view_dd[i, j] = sum(val * ret_ampl["d"][idx]
                                        for idx, val in bas2)

            out[n_s:, :n_s] = np.transpose(out[:n_s, n_s:])
        return out
