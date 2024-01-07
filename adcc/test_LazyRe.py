#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2024 by the adcc authors
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
from .misc import expand_test_templates
from .testdata.cache import cache, psi4_data
from . import backends
from .LazyRe import LazyRe
from .backends import import_scf_results
from .testdata import static_data

import unittest
import pytest
from numpy.testing import assert_allclose


# use the same test cases as for LazyMP
testcases = ["h2o_sto3g", "cn_sto3g"]
if cache.mode_full:
    testcases += ["h2o_def2tzvp", "cn_ccpvdz"]


@expand_test_templates(testcases)
class TestLazyRe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.re = {}
        for case in testcases:
            cls.re[case] = LazyRe(cache.refstate[case], remp_conv_tol=1e-15)

    def template_re2_energy(self, case):
        # 1) compare against the dumped psi4 reference data
        if "psi4" in backends.available():
            ref_data = psi4_data[f"{case}_0_remp2"]
            mol, basis = case.split("_")
            if mol == "cn":
                hf = run_psi4_scf(static_data.xyz[mol], basis, multiplicity=2)
            else:
                hf = run_psi4_scf(static_data.xyz[mol], basis)

            assert_allclose(hf.energy_scf, ref_data["energy_scf"], atol=1e-12)
            re = LazyRe(hf, remp_conv_tol=1e-15)
            assert_allclose(re.energy(2), ref_data["energy_remp"], atol=1e-12)
            assert_allclose(re.energy(2), ref_data["remp_adcc_energy"], atol=1e-12)

        # 2) consistency test against cached adcc data
        ref_data = cache.adcc_reference_data[case]["re"]
        re = self.re[case]
        assert re.energy_correction(2) == pytest.approx(ref_data["re2"]["energy"])

    def template_re3_energy(self, case):
        re = self.re[case]
        ref_data = cache.adcc_reference_data[case]["re"]
        assert re.energy_correction(3) == pytest.approx(ref_data["re3"]["energy"])

    def template_t2(self, case):
        re = self.re[case]
        ref_data = cache.adcc_reference_data[case]["re"]
        assert_allclose(re.t2oo.to_ndarray(), ref_data["re1"]["t_o1o1v1v1"],
                        atol=1e-12)

    def template_td2(self, case):
        re = self.re[case]
        with pytest.raises(RuntimeError):
            re.td2("o1o1v1v1")
        # ref_data = cache.adcc_reference_data[case]["re"]
        # assert_allclose(re.td2("o1o1v1v1").to_ndarray(),
        #                 ref_data["re2"]["td_o1o1v1v1"], atol=1e-12)

    def template_re2_density_mo(self, case):
        mp2diff = self.re[case].mp2_diffdm
        ref_data = cache.adcc_reference_data[case]["re"]

        assert mp2diff.is_symmetric
        for label in ["o1o1", "o1v1", "v1v1"]:
            assert_allclose(mp2diff[label].to_ndarray(),
                            ref_data["re2"]["dm_" + label], atol=1e-12)
        assert "mp2_diffdm" in self.re[case].timer.tasks

    def template_re2_density_ao(self, case):
        mp2diff = self.re[case].mp2_diffdm
        ref_data = cache.adcc_reference_data[case]["re"]
        reference_state = self.re[case].reference_state

        dm_α, dm_β = mp2diff.to_ao_basis(reference_state)
        assert_allclose(dm_α.to_ndarray(), ref_data["re2"]["dm_bb_a"], atol=1e-12)
        assert_allclose(dm_β.to_ndarray(), ref_data["re2"]["dm_bb_b"], atol=1e-12)


def run_psi4_scf(xyz, basis, charge=0, multiplicity=1, conv_tol=1e-12,
                 conv_tol_grad=1e-11, max_iter=150):
    import psi4

    psi4.core.clean_options()
    mol = psi4.geometry(f"""
        {charge} {multiplicity}
        {xyz}
        units au
        symmetry c1
    """)

    basis_sets = {
        "sto3g": "sto-3g",
        "def2tzvp": "def2-tzvp",
        "ccpvdz": "cc-pvdz",
    }
    psi4.set_options({
        'basis': basis_sets[basis],
        'scf_type': "pK",
        'e_convergence': conv_tol,
        'd_convergence': conv_tol_grad,
        'maxiter': max_iter,
        'reference': "RHF",
    })
    if multiplicity != 1:
        psi4.set_options({
            'reference': "UHF",
            'maxiter': max_iter + 500,
            'soscf': 'true'
        })
    psi4.core.be_quiet()
    _, wfn = psi4.energy("scf", return_wfn=True, molecule=mol)
    psi4.core.clean()
    return import_scf_results(wfn)
