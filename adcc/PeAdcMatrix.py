from adcc import AdcMatrix
from .OneParticleOperator import OneParticleOperator


class PeAdcMatrix(AdcMatrix):
    def matvec(self, in_ampl):
        out_ampl = super().matvec(in_ampl)
        operators = self.reference_state.operators
        tdm = OneParticleOperator(self.ground_state, is_symmetric=False)
        tdm.vo = in_ampl.ph.transpose()
        vpe = operators.density_dependent_operators["pe_induction_elec"](tdm)
        # TODO: out_ampl.ph += does not seem to work?!
        out_ampl['ph'] += vpe.ov
        return out_ampl


# TODO: replace with existing shifted matrix
class PeShiftedMat(PeAdcMatrix):
    omega = 0

    def matvec(self, in_ampl):
        out_ampl = super().matvec(in_ampl)
        return out_ampl - in_ampl * self.omega
