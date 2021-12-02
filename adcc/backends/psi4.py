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
import numpy as np
from numpy.testing._private.utils import assert_allclose

from libadcc import HartreeFockProvider
from adcc.misc import cached_property

import psi4

from .EriBuilder import EriBuilder
from ..exceptions import InvalidReference
from ..ExcitedStates import EnergyCorrection


class Psi4OperatorIntegralProvider:
    def __init__(self, wfn):
        self.wfn = wfn
        self.backend = "psi4"
        self.mints = psi4.core.MintsHelper(self.wfn)

    @cached_property
    def electric_dipole(self):
        return [-1.0 * np.asarray(comp) for comp in self.mints.ao_dipole()]

    @cached_property
    def magnetic_dipole(self):
        # TODO: Gauge origin?
        return [
            0.5 * np.asarray(comp)
            for comp in self.mints.ao_angular_momentum()
        ]

    @cached_property
    def nabla(self):
        return [-1.0 * np.asarray(comp) for comp in self.mints.ao_nabla()]

    @property
    def pe_induction_elec(self):
        if hasattr(self.wfn, "pe_state"):
            def pe_induction_elec_ao(dm):
                return self.wfn.pe_state.get_pe_contribution(
                    psi4.core.Matrix.from_array(dm.to_ndarray()),
                    elec_only=True
                )[1]
            return pe_induction_elec_ao


class Psi4EriBuilder(EriBuilder):
    def __init__(self, wfn, n_orbs, n_orbs_alpha, n_alpha, n_beta, restricted,
                 unitary):
        self.wfn = wfn
        self.mints = psi4.core.MintsHelper(self.wfn)
        self.u = unitary
        super().__init__(n_orbs, n_orbs_alpha, n_alpha, n_beta, restricted)

    @property
    def coefficients_new(self):
        u = self.u
        U = {}
        for spin in ["a", "b"]:
            U[spin] = np.block([
                [u[spin]["oo"], np.zeros((u["o" + spin], u["v" + spin]))],
                [np.zeros((u["v" + spin], u["o" + spin])), u[spin]["vv"]]
            ])
            assert_allclose(np.identity(u["o" + spin] + u["v" + spin]),
                            U[spin].T @ U[spin], atol=1e-15)

        occ_a = psi4.core.Matrix(u["oa"] + u["va"], u["oa"])
        virt_a = psi4.core.Matrix(u["oa"] + u["va"], u["va"])
        return {
            "Oa": occ_a.from_array(U["a"][:, :u["oa"]]),
            "Ob": None,  # self.wfn.Cb_subset("AO", "OCC"),
            "Va": virt_a.from_array(U["a"][:, u["oa"]:]),
            "Vb": None  # self.wfn.Cb_subset("AO", "VIR"),
        }

    @property
    def coefficients(self):
        return {
            "Oa": [self.wfn.Ca_subset("AO", "OCC"), self.u["a"]["oo"]],
            "Ob": [self.wfn.Cb_subset("AO", "OCC"), self.u["b"]["oo"]],
            "Va": [self.wfn.Ca_subset("AO", "VIR"), self.u["a"]["vv"]],
            "Vb": [self.wfn.Cb_subset("AO", "VIR"), self.u["b"]["vv"]],
        }

    def compute_mo_eri(self, blocks, spins):
        # print(f"import MO ERI of block {blocks} and spin {spins}")
        # something is wrong here... probably in coefficients_new
        # or I can't use the mo_eri function
        coeffs = tuple(
            self.coefficients[blocks[i] + spins[i]][0] for i in range(4)
        )
        print(f"eri blocks: {blocks}, spins: {spins}")
        eri_mo = np.asarray(self.mints.mo_eri(*coeffs))
        print(f"eri block shape: {eri_mo.shape}")

        # transform mo eri with the unitary matrix
        unitary = tuple(
            self.coefficients[blocks[i] + spins[i]][1] for i in range(4)
        )
        eri_mo = np.einsum("up,vq,uvol,or,ls->pqrs",
                           unitary[0], unitary[1], eri_mo,
                           unitary[2].T, unitary[3].T
                           )
        print(f"transformed eri block shape: {eri_mo.shape}")
        return eri_mo

    def compute_mo_eri_new(self, blocks, spins, fromslices):
        print(f"import MO ERI of block {blocks} and spin {spins}")
        slices = []
        for idx, b in enumerate(blocks):
            if b == "O":
                slices.append(fromslices[idx])
            elif b == "V":
                slices.append(slice(
                    self.n_alpha, self.n_alpha + fromslices[idx].stop
                ))
        slices = tuple(slices)
        print(f"original slices: {fromslices}")
        print(f"adjusted virtual slices: {slices}")
        return None


class Psi4HFProvider(HartreeFockProvider):
    """
        This implementation is only valid
        if no orbital reordering is required.
    """
    def __init__(self, wfn):
        # Do not forget the next line,
        # otherwise weird errors result
        super().__init__()
        self.wfn = wfn
        self.unitary = self.random_unitary()
        self.eri_builder = Psi4EriBuilder(self.wfn, self.n_orbs, self.wfn.nmo(),
                                          wfn.nalpha(), wfn.nbeta(),
                                          self.restricted, self.unitary)
        self.operator_integral_provider = Psi4OperatorIntegralProvider(self.wfn)

    def pe_energy(self, dm, elec_only=True):
        density_psi = psi4.core.Matrix.from_array(dm.to_ndarray())
        e_pe, _ = self.wfn.pe_state.get_pe_contribution(density_psi,
                                                        elec_only=elec_only)
        return e_pe

    @property
    def excitation_energy_corrections(self):
        ret = []
        if self.environment == "pe":
            ptlr = EnergyCorrection(
                "pe_ptlr_correction",
                lambda view: 2.0 * self.pe_energy(view.transition_dm_ao,
                                                  elec_only=True)
            )
            ptss = EnergyCorrection(
                "pe_ptss_correction",
                lambda view: self.pe_energy(view.state_diffdm_ao,
                                            elec_only=True)
            )
            ret.extend([ptlr, ptss])
        return {ec.name: ec for ec in ret}

    @property
    def environment(self):
        ret = None
        if hasattr(self.wfn, "pe_state"):
            ret = "pe"
        return ret

    def get_backend(self):
        return "psi4"

    def get_conv_tol(self):
        conv_tol = psi4.core.get_option("SCF", "E_CONVERGENCE")
        # RMS value of the orbital gradient
        conv_tol_grad = psi4.core.get_option("SCF", "D_CONVERGENCE")
        threshold = max(10 * conv_tol, conv_tol_grad)
        return threshold

    def get_restricted(self):
        return isinstance(self.wfn, (psi4.core.RHF, psi4.core.ROHF))

    def get_energy_scf(self):
        return self.wfn.energy()

    def get_spin_multiplicity(self):
        return self.wfn.molecule().multiplicity()

    def get_n_orbs_alpha(self):
        return self.wfn.nmo()

    def get_n_bas(self):
        return self.wfn.basisset().nbf()

    def get_nuclear_multipole(self, order):
        molecule = self.wfn.molecule()
        if order == 0:
            # The function interface needs to be a np.array on return
            return np.array([sum(molecule.charge(i)
                                 for i in range(molecule.natom()))])
        elif order == 1:
            dip_nuclear = molecule.nuclear_dipole()
            return np.array([dip_nuclear[0], dip_nuclear[1], dip_nuclear[2]])
        else:
            raise NotImplementedError("get_nuclear_multipole with order > 1")

    def fill_orbcoeff_fb(self, out):
        mo_coeff_a = np.asarray(self.wfn.Ca())
        mo_coeff_b = np.asarray(self.wfn.Cb())
        mo_coeff = (mo_coeff_a, mo_coeff_b)
        out[:] = np.transpose(
            np.hstack((mo_coeff[0], mo_coeff[1]))
        )

    def fill_occupation_f(self, out):
        out[:] = np.hstack((
            np.asarray(self.wfn.occupation_a()),
            np.asarray(self.wfn.occupation_b())
        ))

    def fill_orben_f(self, out):
        orben_a = np.asarray(self.wfn.epsilon_a())
        orben_b = np.asarray(self.wfn.epsilon_b())
        out[:] = np.hstack((orben_a, orben_b))

    def fill_fock_ff(self, slices, out):
        u = self.unitary
        U = {}
        diagonal = np.empty(self.n_orbs)
        self.fill_orben_f(diagonal)
        # out[:] = np.diag(diagonal)[slices]
        # build individual blocks
        U["oo"] = np.block([
            [u["a"]["oo"], np.zeros((u["oa"], u["ob"]))],
            [np.zeros((u["ob"], u["oa"])), u["b"]["oo"]]
        ])
        U["vv"] = np.block([
            [u["a"]["vv"], np.zeros((u["va"], u["vb"]))],
            [np.zeros((u["vb"], u["va"])), u["b"]["vv"]]
        ])
        U["ov"] = np.zeros((u["oa"] + u["ob"], u["va"] + u["vb"]))
        U["vo"] = np.zeros((u["va"] + u["vb"], u["oa"] + u["ob"]))

        # build total matrix
        U = np.block([
            [U["oo"], U["ov"]],
            [U["vo"], U["vv"]]
        ])
        assert_allclose(np.identity(U.shape[0]), U.T @ U, atol=1e-15)
        out[:] = (U @ np.diag(diagonal) @ U.T)[slices]
        # print("Import of a Fock Matrix block:")
        # print(f"shape: {np.shape(out)}, slices: {slices}")

    def fill_eri_ffff(self, slices, out):
        self.eri_builder.fill_slice_symm(slices, out)

    def fill_eri_phys_asym_ffff(self, slices, out):
        raise NotImplementedError("fill_eri_phys_asym_ffff not implemented.")

    def has_eri_phys_asym_ffff(self):
        return False

    def flush_cache(self):
        self.eri_builder.flush_cache()

    def orthogonalize_AO(self):
        S = self.wfn.S().to_array()
        s, U = np.linalg.eig(S)
        s_12 = s ** -0.5
        X = U @ np.diag(s_12) @ np.transpose(U)
        return X

    def random_unitary(self):
        from scipy.stats import ortho_group
        from random import choice
        U = {'a': {}, 'b': {}}
        U["oa"] = oa = self.wfn.nalpha()
        U["ob"] = ob = self.wfn.nbeta()
        U["va"] = va = self.wfn.basisset().nbf() - oa
        U["vb"] = vb = self.wfn.basisset().nbf() - ob

        # scheint ein Problem mit dem guess oder davidson zu geben, wenn nur ein
        # occ orbital vorhanden ist... nur 1 der 3 möglichen Single Zustände ist
        # korrekt (der niedrigste). Die anderen hängen von U ab -> random ergebnisse
        # für H2 -2 1 (singlet doppelt negativ) sind ergebnisse konsistent
        U["a"]["oo"] = ortho_group.rvs(oa) if oa > 1 \
            else np.array([[choice((-1, 1))]])
        U["a"]["vv"] = ortho_group.rvs(va) if va > 1 \
            else np.array([[choice((-1, 1))]])
        U["b"]["oo"] = U["a"]["oo"] if self.restricted else ortho_group.rvs(ob)
        U["b"]["vv"] = U["a"]["vv"] if self.restricted else ortho_group.rvs(vb)
        return U


def import_scf(wfn):
    if not isinstance(wfn, psi4.core.HF):
        raise InvalidReference(
            "Only psi4.core.HF and its subtypes are supported references in "
            "backends.psi4.import_scf. This indicates that you passed an "
            "unsupported SCF reference. Make sure you did a restricted or "
            "unrestricted HF calculation."
        )

    if not isinstance(wfn, (psi4.core.RHF, psi4.core.UHF)):
        raise InvalidReference("Right now only RHF and UHF references are "
                               "supported for Psi4.")

    # TODO This is not fully correct, because the core.Wavefunction object
    #      has an internal, but py-invisible Options structure, which contains
    #      the actual set of options ... theoretically they could differ
    scf_type = psi4.core.get_global_option('SCF_TYPE')
    # CD = Choleski, DF = density-fitting
    unsupported_scf_types = ["CD", "DISK_DF", "MEM_DF"]
    if scf_type in unsupported_scf_types:
        raise InvalidReference("Unsupported Psi4 SCF_TYPE, should not be one "
                               f"of {unsupported_scf_types}")

    if wfn.nirrep() > 1:
        raise InvalidReference("The passed Psi4 wave function object needs to "
                               "have exactly one irrep, i.e. be of C1 symmetry.")

    # Psi4 throws an exception if SCF is not converged, so there is no need
    # to assert that here.
    provider = Psi4HFProvider(wfn)
    return provider


def run_hf(xyz, basis, charge=0, multiplicity=1, conv_tol=1e-11,
           conv_tol_grad=1e-8, max_iter=150, pe_options=None):
    basissets = {
        "sto3g": "sto-3g",
        "def2tzvp": "def2-tzvp",
        "ccpvdz": "cc-pvdz",
    }

    mol = psi4.geometry(f"""
        {charge} {multiplicity}
        {xyz}
        symmetry c1
        units au
        no_reorient
        no_com
    """)

    psi4.core.be_quiet()
    psi4.set_options({
        'basis': basissets.get(basis, basis),
        'scf_type': 'pk',
        'e_convergence': conv_tol,
        'd_convergence': conv_tol_grad,
        'maxiter': max_iter,
        'reference': "RHF",
    })
    if pe_options:
        psi4.set_options({"pe": "true"})
        psi4.set_module_options("pe", {"potfile": pe_options["potfile"]})

    if multiplicity != 1:
        psi4.set_options({
            'reference': "UHF",
            'maxiter': max_iter + 500,
            'soscf': 'true'
        })

    _, wfn = psi4.energy('SCF', return_wfn=True, molecule=mol)
    psi4.core.clean()
    return wfn
