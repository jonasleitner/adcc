#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the adcc authors
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
from math import sqrt

from adcc import block as b
from adcc.LazyMp import LazyMp
from adcc.AdcMethod import AdcMethod
from adcc.functions import einsum
from adcc.Intermediates import Intermediates
from adcc.AmplitudeVector import AmplitudeVector
from adcc.OneParticleOperator import OneParticleOperator

from .util import check_doubles_amplitudes, check_singles_amplitudes


def tdm_adc0(mp, amplitude, intermediates):
    # C is either c(ore) or o(ccupied)
    C = b.c if mp.has_core_occupied_space else b.o
    check_singles_amplitudes([C, b.v], amplitude)
    u1 = amplitude.ph

    # Transition density matrix for (CVS-)ADC(0)
    dm = OneParticleOperator(mp, is_symmetric=False)
    dm[b.v + C] = u1.transpose()
    return dm


def tdm_adc1(mp, amplitude, intermediates):
    dm = tdm_adc0(mp, amplitude, intermediates)  # Get ADC(0) result
    # adc1_dp0_ov
    dm.ov = -einsum("ijab,jb->ia", mp.t2(b.oovv), amplitude.ph)
    return dm


def tdm_cvs_adc2(mp, amplitude, intermediates):
    # Get CVS-ADC(1) result (same as CVS-ADC(0))
    dm = tdm_adc0(mp, amplitude, intermediates)
    check_doubles_amplitudes([b.o, b.c, b.v, b.v], amplitude)
    u1 = amplitude.ph
    u2 = amplitude.pphh

    t2 = mp.t2(b.oovv)
    p0 = intermediates.cvs_p0

    # Compute CVS-ADC(2) tdm
    dm.oc = (  # cvs_adc2_dp0_oc
        - einsum("ja,Ia->jI", p0.ov, u1)
        + (1 / sqrt(2)) * einsum("kIab,jkab->jI", u2, t2)
    )

    # cvs_adc2_dp0_vc
    dm.vc -= 0.5 * einsum("ab,Ib->aI", p0.vv, u1)
    return dm


def tdm_adc2(mp, amplitude, intermediates):
    dm = tdm_adc1(mp, amplitude, intermediates)  # Get ADC(1) result
    check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude)
    u1 = amplitude.ph
    u2 = amplitude.pphh

    t2 = mp.t2(b.oovv)
    td2 = mp.td2(b.oovv)
    p0 = mp.mp2_diffdm

    # Compute ADC(2) tdm
    dm.oo = (  # adc2_dp0_oo
        - einsum("ia,ja->ij", p0.ov, u1)
        - einsum("ikab,jkab->ji", u2, t2)
    )
    dm.vv = (  # adc2_dp0_vv
        + einsum("ia,ib->ab", u1, p0.ov)
        + einsum("ijac,ijbc->ab", u2, t2)
    )
    dm.ov -= einsum("ijab,jb->ia", td2, u1)  # adc2_dp0_ov
    dm.vo += 0.5 * (  # adc2_dp0_vo
        + einsum("ijab,jkbc,kc->ai", t2, t2, u1)
        - einsum("ab,ib->ai", p0.vv, u1)
        + einsum("ja,ij->ai", u1, p0.oo)
    )
    return dm


def tdm_adc3(mp, amplitude, intermediates):
    dm = OneParticleOperator(mp, is_symmetric=False)

    ul1, ul2 = amplitude.ph, amplitude.pphh

    t2_1 = mp.t2(b.oovv)
    t2_2 = mp.td2(b.oovv)
    t3_2 = mp.tt2(b.ooovvv)
    t2_3 = mp.td3(b.oovv)

    p0 = mp.mp3_diffdm  # 2nd + 3rd order MP density contribution
    p0_oo, p0_ov, p0_vv = p0.oo, p0.ov, p0.vv
    p0_2 = mp.mp2_diffdm  # 2nd order MP density contribution
    p0_2_oo, p0_2_ov, p0_2_vv = p0_2.oo, p0_2.ov, p0_2.vv

    # The scaling in the comments is given as: [comp_scaling] / [mem_scaling]
    dm.oo = (
        - 1 * einsum('ja,ia->ij', ul1, p0_ov)  # N^3: O^2V^1 / N^2: O^1V^1
        - 1 * einsum('jkab,ikab->ij', ul2, t2_1)  # N^5: O^3V^2 / N^4: O^2V^2
        - 1 * einsum('jkab,ikab->ij', ul2, t2_2)  # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum('jkbc,ikbc->ij', t2_1,  # N^6: O^3V^3 / N^6: O^3V^3
                       einsum('la,iklabc->ikbc', ul1, t3_2))
        - 1 * einsum('jb,ib->ij', p0_2_ov,
                     einsum('ka,ikab->ib', ul1, t2_1))  # N^4: O^2V^2 / N^4: O^2V^2
    )
    dm.ov = (
        - 1 * einsum('jb,ijab->ia', ul1, t2_1)  # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum('jb,ijab->ia', ul1, t2_2)  # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum('jb,ijab->ia', ul1, t2_3)  # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum('jkbc,ijkabc->ia', ul2, t3_2)  # N^6: O^3V^3 / N^6: O^3V^3
        + 1 * einsum('ik,ka->ia', p0_2_oo,
                     einsum('jb,jkab->ka', ul1, t2_1))  # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum('ijac,jc->ia', t2_1,  # N^4: O^2V^2 / N^4: O^2V^2
                       einsum('jb,bc->jc', ul1, p0_2_vv))
        - 1 * einsum('ac,ic->ia', p0_2_vv,
                     einsum('jb,ijbc->ic', ul1, t2_1))  # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum('ikab,kb->ia', t2_1,  # N^4: O^2V^2 / N^4: O^2V^2
                       einsum('jb,jk->kb', ul1, p0_2_oo))
        + 1 * einsum('jlad,ijld->ia', t2_1,  # N^6: O^4V^2 / N^4: O^2V^2
                     einsum('klcd,ijkc->ijld', t2_1,
                            einsum('jb,ikbc->ijkc', ul1, t2_1)))
        + 0.5 * einsum('ilac,lc->ia', t2_1,  # N^4: O^2V^2 / N^4: O^2V^2
                       einsum('klcd,kd->lc', t2_1,
                              einsum('jb,jkbd->kd', ul1, t2_1)))
        - 0.25 * einsum('jkla,ijkl->ia',  # N^6: O^4V^2 / N^4: O^2V^2
                        einsum('jb,klab->jkla', ul1, t2_1),
                        einsum('ijcd,klcd->ijkl', t2_1, t2_1))
    )
    dm.vo = (
        + 0.5 * einsum('ja,ij->ai', ul1, p0_oo)  # N^3: O^2V^1 / N^2: O^1V^1
        - 0.5 * einsum('ib,ab->ai', ul1, p0_vv)  # N^3: O^1V^2 / N^2: V^2
        - 0.5 * einsum('ikab,kb->ai', t2_1,  # N^4: O^2V^2 / N^4: O^2V^2
                       einsum('jc,jkbc->kb', ul1, t2_1))
        - 0.5 * einsum('ikab,kb->ai', t2_1,  # N^4: O^2V^2 / N^4: O^2V^2
                       einsum('jc,jkbc->kb', ul1, t2_2))
        - 0.5 * einsum('ikab,kb->ai', t2_2,  # N^4: O^2V^2 / N^4: O^2V^2
                       einsum('jc,jkbc->kb', ul1, t2_1))
        + 1 * einsum('ia->ai', ul1)  # N^2: O^1V^1 / N^2: O^1V^1
    )
    dm.vv = (
        + 1 * einsum('ia,ib->ab', ul1, p0_ov)  # N^3: O^1V^2 / N^2: V^2
        + 1 * einsum('ijac,ijbc->ab', ul2, t2_1)  # N^5: O^2V^3 / N^4: O^2V^2
        + 1 * einsum('ijac,ijbc->ab', ul2, t2_2)  # N^5: O^2V^3 / N^4: O^2V^2
        + 1 * einsum('ja,jb->ab', p0_2_ov,
                     einsum('ic,ijbc->jb', ul1, t2_1))  # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum('jkad,jkbd->ab', t2_1,  # N^6: O^3V^3 / N^6: O^3V^3
                       einsum('ic,ijkbcd->jkbd', ul1, t3_2))
    )
    return dm


DISPATCH = {
    "adc0": tdm_adc0,
    "adc1": tdm_adc1,
    "adc2": tdm_adc2,
    "adc2x": tdm_adc2,
    "adc3": tdm_adc3,
    "cvs-adc0": tdm_adc0,
    "cvs-adc1": tdm_adc0,  # No extra contribs for CVS-ADC(1)
    "cvs-adc2": tdm_cvs_adc2,
    "cvs-adc2x": tdm_cvs_adc2,
}


def transition_dm(method, ground_state, amplitude, intermediates=None):
    """
    Compute the one-particle transition density matrix from ground to excited
    state in the MO basis.

    Parameters
    ----------
    method : str, AdcMethod
        The method to use for the computation (e.g. "adc2")
    ground_state : LazyMp
        The ground state upon which the excitation was based
    amplitude : AmplitudeVector
        The amplitude vector
    intermediates : adcc.Intermediates
        Intermediates from the ADC calculation to reuse
    """
    if not isinstance(method, AdcMethod):
        method = AdcMethod(method)
    if not isinstance(ground_state, LazyMp):
        raise TypeError("ground_state should be a LazyMp object.")
    if not isinstance(amplitude, AmplitudeVector):
        raise TypeError("amplitude should be an AmplitudeVector object.")
    if intermediates is None:
        intermediates = Intermediates(ground_state)

    if method.name not in DISPATCH:
        raise NotImplementedError("transition_dm is not implemented "
                                  f"for {method.name}.")
    else:
        ret = DISPATCH[method.name](ground_state, amplitude, intermediates)
        return ret.evaluate()
