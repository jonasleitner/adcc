import adcc
import pyscf
from pyscf.solvent import ddCOSMO
from pyscf import scf
from adcc.testdata.cache import psi4_data

basis = "sto-3g"

mol = pyscf.M(
    atom="""
    C 2.0092420208996 3.8300915804899 0.8199294419789
    O 2.1078857690998 2.0406638776593 2.1812021228452
    H 2.0682421748693 5.7438044586615 1.5798996515014
    H 1.8588483602149 3.6361694243085 -1.2192956060942
     """,
    basis=basis, symmetry=0, charge=0, spin=0, verbose=3,
    unit="Bohr"
)

# for comparison with psi4 data use pcmsolver eps
mf = ddCOSMO(scf.RHF(mol))
mf.with_solvent.eps = 78.39
mf.conv_tol = 1e-11
mf.conv_tol_grad = 1e-10
mf.max_cycle = 150

mf.kernel()

mf.with_solvent.eps = 1.776
auto = adcc.adc1(mf, n_singlets=5, conv_tol=1e-7,
                 environment="linear_response")
print(auto.describe())


name = f"formaldehyde_{basis}_pcm_adc1"

diff = mf.e_tot - psi4_data[name]["energy_scf"]
print(f"difference to psi4 scf energy [Eh]: {diff}")

diff = auto.excitation_energy_uncorrected - \
       psi4_data[name]["excitation_energy"]
print(f"difference to psi4 ex energies [Eh]: {diff}")
