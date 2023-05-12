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


def diffdm_adc0(mp, amplitude, intermediates):
    # C is either c(ore) or o(ccupied)
    C = b.c if mp.has_core_occupied_space else b.o
    check_singles_amplitudes([C, b.v], amplitude)
    u1 = amplitude.ph

    dm = OneParticleOperator(mp, is_symmetric=True)
    dm[C + C] = -einsum("ia,ja->ij", u1, u1)
    dm.vv = einsum("ia,ib->ab", u1, u1)
    return dm


def diffdm_adc2(mp, amplitude, intermediates):
    dm = diffdm_adc0(mp, amplitude, intermediates)  # Get ADC(1) result
    check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude)
    u1, u2 = amplitude.ph, amplitude.pphh

    t2 = mp.t2(b.oovv)
    p0 = mp.mp2_diffdm
    p1_oo = dm.oo.evaluate()  # ADC(1) diffdm
    p1_vv = dm.vv.evaluate()  # ADC(1) diffdm

    # Zeroth order doubles contributions
    p2_oo = -einsum("ikab,jkab->ij", u2, u2)
    p2_vv = einsum("ijac,ijbc->ab", u2, u2)
    p2_ov = -2 * einsum("jb,ijab->ia", u1, u2).evaluate()

    # ADC(2) ISR intermediate (TODO Move to intermediates)
    ru1 = einsum("ijab,jb->ia", t2, u1).evaluate()

    # Compute second-order contributions to the density matrix
    dm.oo = (  # adc2_p_oo
        p1_oo + 2 * p2_oo - einsum("ia,ja->ij", ru1, ru1) + (
            + einsum("ik,kj->ij", p1_oo, p0.oo)
            - einsum("ikcd,jkcd->ij", t2,
                     + 0.5 * einsum("lk,jlcd->jkcd", p1_oo, t2)
                     - einsum("jkcb,db->jkcd", t2, p1_vv))
            - einsum("ia,jkac,kc->ij", u1, t2, ru1)
        ).symmetrise()
    )

    dm.vv = (  # adc2_p_vv
        p1_vv + 2 * p2_vv + einsum("ia,ib->ab", ru1, ru1) - (
            + einsum("ac,cb->ab", p1_vv, p0.vv)
            + einsum("klbc,klac->ab", t2,
                     + 0.5 * einsum("klad,cd->klac", t2, p1_vv)
                     - einsum("jk,jlac->klac", p1_oo, t2))
            - einsum("ikac,kc,ib->ab", t2, ru1, u1)
        ).symmetrise()
    )

    dm.ov = (  # adc2_p_ov
        + p2_ov
        - einsum("ijab,jb->ia", t2, p2_ov)
        - einsum("ib,ba->ia", p0.ov, p1_vv)
        + einsum("ij,ja->ia", p1_oo, p0.ov)
        - einsum("ib,klca,klcb->ia", u1, t2, u2)
        - einsum("ikcd,jkcd,ja->ia", t2, u2, u1)
    )
    return dm


def diffdm_cvs_adc2(mp, amplitude, intermediates):
    dm = diffdm_adc0(mp, amplitude, intermediates)  # Get ADC(1) result
    check_doubles_amplitudes([b.o, b.c, b.v, b.v], amplitude)
    u1, u2 = amplitude.ph, amplitude.pphh

    t2 = mp.t2(b.oovv)
    p0 = intermediates.cvs_p0
    p1_vv = dm.vv.evaluate()  # ADC(1) diffdm

    # Zeroth order doubles contributions
    p2_ov = -sqrt(2) * einsum("jb,ijab->ia", u1, u2)
    p2_vo = -sqrt(2) * einsum("ijab,jb->ai", u2, u1)
    p2_oo = -einsum("ljab,kjab->kl", u2, u2)
    p2_vv = 2 * einsum("ijac,ijbc->ab", u2, u2)

    # Second order contributions
    # cvs_adc2_dp_oo
    dm.oo = p2_oo + einsum("ab,ikac,jkbc->ij", p1_vv, t2, t2)

    dm.ov = p2_ov + (  # cvs_adc2_dp_ov
        - einsum("ka,ab->kb", p0.ov, p1_vv)
        - einsum("lkdb,dl->kb", t2, p2_vo)
        + 1 / sqrt(2) * einsum("ib,klad,liad->kb", u1, t2, u2)
    )

    dm.vv = p1_vv + p2_vv - 0.5 * (  # cvs_adc2_dp_vv
        + einsum("cb,ac->ab", p1_vv, p0.vv)
        + einsum("cb,ac->ab", p0.vv, p1_vv)
        + einsum("ijbc,ijad,cd->ab", t2, t2, p1_vv)
    )

    # Add 2nd order correction to CVS-ADC(1) diffdm
    dm.cc -= einsum("kIab,kJab->IJ", u2, u2)
    return dm


def diffdm_adc3(mp: LazyMp, amplitude: AmplitudeVector,
                intermediates: Intermediates) -> OneParticleOperator:
    check_singles_amplitudes([b.o, b.v], amplitude)
    check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude)

    dm = OneParticleOperator(mp, is_symmetric=True)

    ur1, ur2 = amplitude.ph, amplitude.pphh

    t2_1 = mp.t2oo
    t2_2 = mp.td2(b.oovv)
    t3_2 = mp.tt2(b.ooovvv)

    p0_2 = mp.mp2_diffdm
    p0_2_oo, p0_2_ov, p0_2_vv = p0_2.oo, p0_2.ov, p0_2.vv
    p0 = mp.mp3_diffdm
    p0_oo, p0_ov, p0_vv = p0.oo, p0.ov, p0.vv

    # NOTE: In the equations below only MP densities have been factored.
    #       They can be further simplified by factoring 0'th order contributions,
    #       ru1 and a similar intermediate with td2 (t2_2)

    # The scaling in the comments is given as: [comp_scaling] / [mem_scaling]
    dm.oo += (
        - 1 * einsum('ia,ja->ij', ur1, ur1)  # N^3: O^2V^1 / N^2: O^1V^1
        - 2 * einsum('ikab,jkab->ij', ur2, ur2)  # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum('jlbc,ilbc->ij', t2_1,  # N^5: O^3V^2 / N^4: O^2V^2
                       einsum('ikbc,kl->ilbc', t2_1,
                              einsum('ka,la->kl', ur1, ur1)))
        - 1 * einsum('jlbc,ilbc->ij', t2_1,  # N^5: O^2V^3 / N^4: O^2V^2
                     einsum('ilab,ac->ilbc', t2_1,
                            einsum('ka,kc->ac', ur1, ur1)))
        - 1 * einsum('ia,ja->ij', einsum('kb,ikab->ia', ur1, t2_1),
                     einsum('lc,jlac->ja', ur1, t2_1))  # N^4: O^2V^2 / N^4: O^2V^2
        + 2 * (  # factor 2: revert the 1/2 from symmetrise
            - 2 * einsum('jb,ib->ij', p0_2_ov,  # N^4: O^2V^2 / N^4: O^2V^2
                         einsum('ka,ikab->ib', ur1, ur2))
            - 0.5 * einsum('jk,ik->ij', p0_oo,  # N^3: O^2V^1 / N^2: O^1V^1
                           einsum('ia,ka->ik', ur1, ur1))
            + 1 * einsum('ib,jb->ij',  # N^4: O^2V^2 / N^4: O^2V^2
                         einsum('ka,ikab->ib', ur1, t2_1),
                         einsum('lc,jlbc->jb', ur1, t2_2))
            + 1 * einsum('jkac,ikac->ij', t2_2,  # N^5: O^2V^3 / N^4: O^2V^2
                         einsum('ikbc,ab->ikac', t2_1,
                                einsum('la,lb->ab', ur1, ur1)))
            + 0.5 * einsum('jlbc,ilbc->ij', t2_2,  # N^5: O^3V^2 / N^4: O^2V^2
                           einsum('ikbc,kl->ilbc', t2_1,
                                  einsum('ka,la->kl', ur1, ur1)))
            + 0.5 * einsum('ic,jc->ij', ur1,  # N^4: O^2V^2 / N^4: O^2V^2
                           einsum('jlbc,lb->jc', t2_1,
                                  einsum('ka,klab->lb', ur1, t2_1)))
            + 0.5 * einsum('ib,jb->ij', ur1,  # N^4: O^2V^2 / N^4: O^2V^2
                           einsum('jkbc,kc->jb', t2_2,
                                  einsum('la,klac->kc', ur1, t2_1)))
            + 0.5 * einsum('ib,jb->ij', ur1,  # N^4: O^2V^2 / N^4: O^2V^2
                           einsum('jkbc,kc->jb', t2_1,
                                  einsum('la,klac->kc', ur1, t2_2)))
        )
    ).symmetrise()
    dm.ov = (
        - 2 * einsum('jb,ijab->ia', ur1, ur2)  # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum('ka,ik->ia', ur1,  # N^5: O^3V^2 / N^4: O^2V^2
                     einsum('jkbc,ijbc->ik', ur2, t2_1))
        + 1 * einsum('ka,ik->ia', ur1,  # N^5: O^3V^2 / N^4: O^2V^2
                     einsum('jkbc,ijbc->ik', ur2, t2_2))
        + 1 * einsum('ic,ac->ia', ur1,  # N^5: O^2V^3 / N^4: O^2V^2
                     einsum('jkbc,jkab->ac', ur2, t2_2))
        + 1 * einsum('ab,ib->ia', p0_2_vv,
                     einsum('jc,ijbc->ib', ur1, ur2))  # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum('ja,ij->ia', ur1,
                     einsum('jb,ib->ij', ur1, p0_ov))  # N^3: O^2V^1 / N^2: O^1V^1
        - 1 * einsum('ja,ij->ia', p0_ov,
                     einsum('ib,jb->ij', ur1, ur1))  # N^3: O^2V^1 / N^2: O^1V^1
        - 1 * einsum('ib,ab->ia', ur1,  # N^5: O^2V^3 / N^4: O^2V^2
                     einsum('jkbc,jkac->ab', ur2, t2_1))
        - 1 * einsum('ij,ja->ia', p0_2_oo,  # N^4: O^2V^2 / N^4: O^2V^2
                     einsum('kb,jkab->ja', ur1, ur2))
        - 2 * einsum('ijac,jc->ia', t2_1,
                     einsum('kb,jkbc->jc', ur1, ur2))  # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * einsum('ikab,kb->ia', t2_2,
                     einsum('jc,jkbc->kb', ur1, ur2))  # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum('ijac,jc->ia', t2_1,  # N^4: O^2V^2 / N^4: O^2V^2
                     einsum('kc,jk->jc', p0_2_ov, einsum('jb,kb->jk', ur1, ur1)))
        + 1 * einsum('ja,ij->ia', einsum('kb,jkab->ja', ur1, t2_1),
                     einsum('ic,jc->ij', ur1, p0_2_ov))  # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum('ijab,jb->ia', t2_1,  # N^4: O^2V^2 / N^4: O^2V^2
                     einsum('jc,bc->jb', p0_2_ov, einsum('kb,kc->bc', ur1, ur1)))
        + 0.5 * einsum('ib,ab->ia', ur1,  # N^6: O^3V^3 / N^6: O^3V^3
                       einsum('klbc,klac->ab', t2_1,
                              einsum('jd,jklacd->klac', ur1, t3_2)))
        + 0.5 * einsum('iklacd,klcd->ia', t3_2,  # N^6: O^3V^3 / N^6: O^3V^3
                       einsum('klbd,bc->klcd', t2_1, einsum('jb,jc->bc', ur1, ur1)))
        + 0.5 * einsum('id,ad->ia',  # N^5: O^2V^3 / N^4: O^2V^2
                       einsum('lb,ilbd->id', ur1, t2_1),
                       einsum('jkac,jkcd->ad', ur2, t2_1))
        + 0.5 * einsum('ikac,kc->ia', t2_1,  # N^5: O^3V^2 / N^4: O^2V^2
                       einsum('jc,jk->kc', ur1, einsum('jlbd,klbd->jk', ur2, t2_1)))
        + 0.5 * einsum('ilad,ld->ia', t2_1,  # N^5: O^2V^3 / N^4: O^2V^2
                       einsum('lc,cd->ld', ur1, einsum('jkbc,jkbd->cd', ur2, t2_1)))
        + 0.5 * einsum('ijkabc,jkbc->ia', t3_2,  # N^6: O^3V^3 / N^6: O^3V^3
                       einsum('jlbc,kl->jkbc', t2_1, einsum('kd,ld->kl', ur1, ur1)))
        + 0.5 * einsum('la,il->ia',  # N^5: O^3V^2 / N^4: O^2V^2
                       einsum('kd,klad->la', ur1, t2_1),
                       einsum('ijbc,jlbc->il', ur2, t2_1))
        - 1 * einsum('ac,ic->ia', einsum('ka,kc->ac', ur1, p0_2_ov),
                     einsum('jb,ijbc->ic', ur1, t2_1))  # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum('ijkacd,jkcd->ia', t3_2,  # N^6: O^3V^3 / N^6: O^3V^3
                     einsum('jd,kc->jkcd', ur1, einsum('lb,klbc->kc', ur1, t2_1)))
        - 1 * einsum('jlab,ijlb->ia', ur2,  # N^6: O^4V^2 / N^4: O^2V^2
                     einsum('klbd,ijkd->ijlb', t2_1,
                            einsum('jc,ikcd->ijkd', ur1, t2_1)))
        - 1 * einsum('ikac,kc->ia', t2_1,  # N^4: O^2V^2 / N^4: O^2V^2
                     einsum('klbc,lb->kc', t2_1, einsum('jd,jlbd->lb', ur1, ur2)))
        - 1 * einsum('klac,iklc->ia', t2_1,  # N^6: O^4V^2 / N^4: O^2V^2
                     einsum('jlbc,ijkb->iklc', t2_1,
                            einsum('kd,ijbd->ijkb', ur1, ur2)))
        - 0.5 * einsum('ja,ij->ia', ur1,  # N^6: O^3V^3 / N^6: O^3V^3
                       einsum('jlcd,ilcd->ij', t2_1,
                              einsum('kb,iklbcd->ilcd', ur1, t3_2)))
        - 0.25 * einsum('jkla,ijkl->ia',  # N^6: O^4V^2 / N^4: O^2V^2
                        einsum('jb,klab->jkla', ur1, ur2),
                        einsum('ijcd,klcd->ijkl', t2_1, t2_1))
        - 0.25 * einsum('jkla,ijkl->ia',  # N^6: O^4V^2 / N^4: O^2V^2
                        einsum('jb,klab->jkla', ur1, t2_1),
                        einsum('ijcd,klcd->ijkl', ur2, t2_1))
    )
    dm.vv = (
        + 1 * einsum('ia,ib->ab', ur1, ur1)  # N^3: O^1V^2 / N^2: V^2
        + 2 * einsum('ijac,ijbc->ab', ur2, ur2)  # N^5: O^2V^3 / N^4: O^2V^2
        + 1 * einsum('ka,kb->ab', einsum('ic,ikac->ka', ur1, t2_1),
                     einsum('jd,jkbd->kb', ur1, t2_1))  # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum('jkbd,jkad->ab', t2_1,  # N^5: O^2V^3 / N^4: O^2V^2
                     einsum('ikad,ij->jkad', t2_1, einsum('ic,jc->ij', ur1, ur1)))
        - 0.5 * einsum('jkbd,jkad->ab', t2_1,  # N^5: O^2V^3 / N^4: O^2V^2
                       einsum('jkac,cd->jkad', t2_1, einsum('ic,id->cd', ur1, ur1)))
        + 2 * (  # factor 2: revert the 1/2 from symmetrise
            + 2 * einsum('jb,ja->ab', p0_2_ov,  # N^4: O^2V^2 / N^4: O^2V^2
                         einsum('ic,ijac->ja', ur1, ur2))
            - 0.5 * einsum('ia,ib->ab', ur1,  # N^3: O^1V^2 / N^2: V^2
                           einsum('ic,bc->ib', ur1, p0_vv))
            + 1 * einsum('jkbc,jkac->ab', t2_2,  # N^5: O^2V^3 / N^4: O^2V^2
                         einsum('ijac,ik->jkac', t2_1,
                                einsum('id,kd->ik', ur1, ur1)))
            + 0.5 * einsum('ia,ib->ab', ur1,  # N^4: O^2V^2 / N^4: O^2V^2
                           einsum('ikbd,kd->ib', t2_1,
                                  einsum('jc,jkcd->kd', ur1, t2_1)))
            + 0.5 * einsum('ja,jb->ab', ur1,  # N^4: O^2V^2 / N^4: O^2V^2
                           einsum('jkbd,kd->jb', t2_1,
                                  einsum('ic,ikcd->kd', ur1, t2_2)))
            + 0.5 * einsum('ja,jb->ab', ur1,  # N^4: O^2V^2 / N^4: O^2V^2
                           einsum('jkbd,kd->jb', t2_2,
                                  einsum('ic,ikcd->kd', ur1, t2_1)))
            - 1 * einsum('jb,ja->ab',  # N^4: O^2V^2 / N^4: O^2V^2
                         einsum('ic,ijbc->jb', ur1, t2_2),
                         einsum('kd,jkad->ja', ur1, t2_1))
            - 0.5 * einsum('jkbc,jkac->ab', t2_2,  # N^5: O^2V^3 / N^4: O^2V^2
                           einsum('jkad,cd->jkac', t2_1,
                                  einsum('ic,id->cd', ur1, ur1)))
        )
    ).symmetrise()
    return dm


# dict controlling the dispatch of the state_diffdm function
DISPATCH = {
    "adc0": diffdm_adc0,
    "adc1": diffdm_adc0,       # same as ADC(0)
    "adc2": diffdm_adc2,
    "adc2x": diffdm_adc2,
    "adc3": diffdm_adc3,
    "cvs-adc0": diffdm_adc0,
    "cvs-adc1": diffdm_adc0,   # same as ADC(0)
    "cvs-adc2": diffdm_cvs_adc2,
    "cvs-adc2x": diffdm_cvs_adc2,
}


def state_diffdm(method, ground_state, amplitude, intermediates=None):
    """
    Compute the one-particle difference density matrix of an excited state
    in the MO basis.

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
        raise NotImplementedError("state_diffdm is not implemented "
                                  f"for {method.name}.")
    else:
        ret = DISPATCH[method.name](ground_state, amplitude, intermediates)
        return ret.evaluate()
