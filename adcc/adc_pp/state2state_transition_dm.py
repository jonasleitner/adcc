#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
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
from adcc import block as b
from adcc.LazyMp import LazyMp
from adcc.AdcMethod import AdcMethod
from adcc.functions import einsum
from adcc.Intermediates import Intermediates
from adcc.AmplitudeVector import AmplitudeVector
from adcc.OneParticleOperator import OneParticleOperator

from .util import check_doubles_amplitudes, check_singles_amplitudes


def s2s_tdm_adc0(mp, amplitude_l, amplitude_r, intermediates):
    check_singles_amplitudes([b.o, b.v], amplitude_l, amplitude_r)
    ul1 = amplitude_l.ph
    ur1 = amplitude_r.ph

    dm = OneParticleOperator(mp, is_symmetric=False)
    dm.oo = -einsum('ja,ia->ij', ul1, ur1)
    dm.vv = einsum('ia,ib->ab', ul1, ur1)
    return dm


def s2s_tdm_adc2(mp, amplitude_l, amplitude_r, intermediates):
    check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude_l, amplitude_r)
    dm = s2s_tdm_adc0(mp, amplitude_l, amplitude_r, intermediates)

    ul1, ul2 = amplitude_l.ph, amplitude_l.pphh
    ur1, ur2 = amplitude_r.ph, amplitude_r.pphh

    t2 = mp.t2(b.oovv)
    p0 = mp.mp2_diffdm
    p1_oo = dm.oo.evaluate()  # ADC(1) tdm
    p1_vv = dm.vv.evaluate()  # ADC(1) tdm

    # ADC(2) ISR intermediate (TODO Move to intermediates)
    rul1 = einsum('ijab,jb->ia', t2, ul1).evaluate()
    rur1 = einsum('ijab,jb->ia', t2, ur1).evaluate()

    dm.oo = (
        p1_oo - 2.0 * einsum('ikab,jkab->ij', ur2, ul2)
        + 0.5 * einsum('ik,kj->ij', p1_oo, p0.oo)
        + 0.5 * einsum('ik,kj->ij', p0.oo, p1_oo)
        - 0.5 * einsum('ikcd,lk,jlcd->ij', t2, p1_oo, t2)
        + 1.0 * einsum('ikcd,jkcb,db->ij', t2, t2, p1_vv)
        - 0.5 * einsum('ia,jkac,kc->ij', ur1, t2, rul1)
        - 0.5 * einsum('ikac,kc,ja->ij', t2, rur1, ul1)
        - 1.0 * einsum('ia,ja->ij', rul1, rur1)
    )
    dm.vv = (
        p1_vv + 2.0 * einsum('ijac,ijbc->ab', ul2, ur2)
        - 0.5 * einsum("ac,cb->ab", p1_vv, p0.vv)
        - 0.5 * einsum("ac,cb->ab", p0.vv, p1_vv)
        - 0.5 * einsum("klbc,klad,cd->ab", t2, t2, p1_vv)
        + 1.0 * einsum("klbc,jk,jlac->ab", t2, p1_oo, t2)
        + 0.5 * einsum("ikac,kc,ib->ab", t2, rul1, ur1)
        + 0.5 * einsum("ia,ikbc,kc->ab", ul1, t2, rur1)
        + 1.0 * einsum("ia,ib->ab", rur1, rul1)
    )

    p1_ov = -2.0 * einsum("jb,ijab->ia", ul1, ur2).evaluate()
    p1_vo = -2.0 * einsum("ijab,jb->ai", ul2, ur1).evaluate()

    dm.ov = (
        p1_ov
        - einsum("ijab,bj->ia", t2, p1_vo)
        - einsum("ib,ba->ia", p0.ov, p1_vv)
        + einsum("ij,ja->ia", p1_oo, p0.ov)
        - einsum("ib,klca,klcb->ia", ur1, t2, ul2)
        - einsum("ikcd,jkcd,ja->ia", t2, ul2, ur1)
    )
    dm.vo = (
        p1_vo
        - einsum("ijab,jb->ai", t2, p1_ov)
        - einsum("ib,ab->ai", p0.ov, p1_vv)
        + einsum("ji,ja->ai", p1_oo, p0.ov)
        - einsum("ib,klca,klcb->ai", ul1, t2, ur2)
        - einsum("ikcd,jkcd,ja->ai", t2, ur2, ul1)
    )
    return dm


def s2s_tdm_adc3(mp: LazyMp, amplitude_l: AmplitudeVector,
                 amplitude_r: AmplitudeVector,
                 intermediates: Intermediates) -> OneParticleOperator:
    check_singles_amplitudes([b.o, b.v], amplitude_l, amplitude_r)
    check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude_l, amplitude_r)

    dm = OneParticleOperator(mp, is_symmetric=False)

    ul1, ul2 = amplitude_l.ph, amplitude_l.pphh
    ur1, ur2 = amplitude_r.ph, amplitude_r.pphh

    t2_1 = mp.t2(b.oovv)
    t2_2 = mp.td2(b.oovv)
    t3_2 = mp.tt2(b.ooovvv)

    p0_2 = mp.mp2_diffdm  # 2nd order contrib
    p0_2_oo, p0_2_ov, p0_2_vv = p0_2.oo, p0_2.ov, p0_2.vv
    p0 = mp.mp3_diffdm  # 2nd + 3rd order contrib
    p0_oo, p0_ov, p0_vv = p0.oo, p0.ov, p0.vv

    # NOTE: only the MP densities have been factored as intermediates.
    #       The equations can be further simplified by factoring:
    #       - the zeroth order ADC density contributions
    #       - ru1 and similar intermediates

    # The scaling in the comments is given as: [comp_scaling] / [mem_scaling]
    dm.oo = (
        - 1 * einsum('ja,ia->ij', ul1, ur1)  # N^3: O^2V^1 / N^2: O^1V^1
        - 2 * einsum('jkab,ikab->ij', ul2, ur2)  # N^5: O^3V^2 / N^4: O^2V^2
        # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * einsum('jb,ib->ij', p0_2_ov, einsum('ka,ikab->ib', ul1, ur2))
        # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * einsum('ib,jb->ij', p0_2_ov, einsum('jkab,ka->jb', ul2, ur1))
        # N^3: O^2V^1 / N^2: O^1V^1
        - 0.5 * einsum('ik,jk->ij', p0_oo, einsum('ja,ka->jk', ul1, ur1))
        # N^3: O^2V^1 / N^2: O^1V^1
        - 0.5 * einsum('jk,ik->ij', p0_oo, einsum('ka,ia->ik', ul1, ur1))
        + 1 * einsum('ib,jb->ij',  # N^4: O^2V^2 / N^4: O^2V^2
                     einsum('ka,ikab->ib', ul1, t2_1),
                     einsum('lc,jlbc->jb', ur1, t2_2))
        + 1 * einsum('ikac,jkac->ij', t2_2,  # N^5: O^2V^3 / N^4: O^2V^2
                     einsum('jkbc,ab->jkac', t2_1, einsum('la,lb->ab', ul1, ur1)))
        + 0.5 * einsum('jlbc,ilbc->ij', t2_1,  # N^5: O^3V^2 / N^4: O^2V^2
                       einsum('ikbc,kl->ilbc', t2_1, einsum('ka,la->kl', ul1, ur1)))
        + 0.5 * einsum('jlbc,ilbc->ij', t2_2,  # N^5: O^3V^2 / N^4: O^2V^2
                       einsum('ikbc,kl->ilbc', t2_1, einsum('ka,la->kl', ul1, ur1)))
        + 0.5 * einsum('ic,jc->ij', ur1,  # N^4: O^2V^2 / N^4: O^2V^2
                       einsum('jlbc,lb->jc', t2_1,
                              einsum('ka,klab->lb', ul1, t2_1)))
        + 0.5 * einsum('ilbc,jlbc->ij', t2_2,  # N^5: O^3V^2 / N^4: O^2V^2
                       einsum('jkbc,kl->jlbc', t2_1, einsum('la,ka->kl', ul1, ur1)))
        + 0.5 * einsum('ib,jb->ij', ur1,  # N^4: O^2V^2 / N^4: O^2V^2
                       einsum('jkbc,kc->jb', t2_2,
                              einsum('la,klac->kc', ul1, t2_1)))
        + 0.5 * einsum('ib,jb->ij', ur1,  # N^4: O^2V^2 / N^4: O^2V^2
                       einsum('jkbc,kc->jb', t2_1,
                              einsum('la,klac->kc', ul1, t2_2)))
        - 1 * einsum('ic,jc->ij',
                     einsum('ka,ikac->ic', ul1, t2_2),
                     einsum('lb,jlbc->jc', ur1, t2_1))  # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum('jlbc,ilbc->ij', t2_1,  # N^5: O^2V^3 / N^4: O^2V^2
                     einsum('ilab,ac->ilbc', t2_1, einsum('ka,kc->ac', ul1, ur1)))
        - 1 * einsum('jlbc,ilbc->ij', t2_2,  # N^5: O^2V^3 / N^4: O^2V^2
                     einsum('ilab,ac->ilbc', t2_1, einsum('ka,kc->ac', ul1, ur1)))
        - 1 * einsum('ia,ja->ij',
                     einsum('kb,ikab->ia', ul1, t2_1),
                     einsum('lc,jlac->ja', ur1, t2_1))  # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum('ja,ia->ij', ul1,  # N^4: O^2V^2 / N^4: O^2V^2
                       einsum('ilac,lc->ia', t2_1,
                              einsum('kb,klbc->lc', ur1, t2_2)))
        - 0.5 * einsum('ja,ia->ij', ul1,  # N^4: O^2V^2 / N^4: O^2V^2
                       einsum('ilac,lc->ia', t2_2,
                              einsum('kb,klbc->lc', ur1, t2_1)))
        - 0.5 * einsum('jb,ib->ij', ul1,  # N^4: O^2V^2 / N^4: O^2V^2
                       einsum('ilbc,lc->ib', t2_1,
                              einsum('ka,klac->lc', ur1, t2_1)))
    )
    dm.ov = (
        - 2 * einsum('jb,ijab->ia', ul1, ur2)  # N^4: O^2V^2 / N^4: O^2V^2
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum('ik,ka->ia', p0_2_oo, einsum('jb,jkab->ka', ul1, ur2))
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum('ab,ib->ia', p0_2_vv, einsum('jc,ijbc->ib', ul1, ur2))
        # N^3: O^2V^1 / N^2: O^1V^1
        - 1 * einsum('ja,ij->ia', ur1, einsum('jb,ib->ij', ul1, p0_ov))
        # N^3: O^2V^1 / N^2: O^1V^1
        - 1 * einsum('ja,ij->ia', p0_ov, einsum('jb,ib->ij', ul1, ur1))
        # N^5: O^3V^2 / N^4: O^2V^2
        - 1 * einsum('ja,ij->ia', ur1, einsum('jkbc,ikbc->ij', ul2, t2_1))
        # N^5: O^3V^2 / N^4: O^2V^2
        - 1 * einsum('ja,ij->ia', ur1, einsum('jkbc,ikbc->ij', ul2, t2_2))
        # N^5: O^2V^3 / N^4: O^2V^2
        - 1 * einsum('ib,ab->ia', ur1, einsum('jkbc,jkac->ab', ul2, t2_1))
        # N^5: O^2V^3 / N^4: O^2V^2
        - 1 * einsum('ib,ab->ia', ur1, einsum('jkbc,jkac->ab', ul2, t2_2))
        # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * einsum('ikab,kb->ia', t2_1, einsum('jkbc,jc->kb', ul2, ur1))
        # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * einsum('ikab,kb->ia', t2_2, einsum('jkbc,jc->kb', ul2, ur1))
        + 1 * einsum('ijac,jc->ia', t2_1,  # N^4: O^2V^2 / N^4: O^2V^2
                     einsum('kc,jk->jc', p0_2_ov, einsum('jb,kb->jk', ul1, ur1)))
        + 1 * einsum('ikab,kb->ia', t2_1,  # N^4: O^2V^2 / N^4: O^2V^2
                     einsum('kc,bc->kb', p0_2_ov, einsum('jb,jc->bc', ul1, ur1)))
        + 1 * einsum('ja,ij->ia',
                     einsum('kb,jkab->ja', ul1, t2_1),
                     einsum('ic,jc->ij', ur1, p0_2_ov))  # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum('la,il->ia', ur1,  # N^6: O^3V^3 / N^6: O^3V^3
                       einsum('klcd,ikcd->il', t2_1,
                              einsum('jb,ijkbcd->ikcd', ul1, t3_2)))
        + 0.5 * einsum('ikab,kb->ia', t2_1,  # N^5: O^3V^2 / N^4: O^2V^2
                       einsum('jb,jk->kb', ul1, einsum('jlcd,klcd->jk', ur2, t2_1)))
        + 0.5 * einsum('id,ad->ia',  # N^5: O^2V^3 / N^4: O^2V^2
                       einsum('lb,ilbd->id', ul1, t2_1),
                       einsum('jkac,jkcd->ad', ur2, t2_1))
        + 0.5 * einsum('iklacd,klcd->ia', t3_2,  # N^6: O^3V^3 / N^6: O^3V^3
                       einsum('klbd,bc->klcd', t2_1, einsum('jc,jb->bc', ul1, ur1)))
        + 0.5 * einsum('ilad,ld->ia', t2_1,  # N^5: O^2V^3 / N^4: O^2V^2
                       einsum('lc,cd->ld', ul1, einsum('jkbc,jkbd->cd', ur2, t2_1)))
        + 0.5 * einsum('ib,ab->ia', ur1,  # N^6: O^3V^3 / N^6: O^3V^3
                       einsum('klbc,klac->ab', t2_1,
                              einsum('jd,jklacd->klac', ul1, t3_2)))
        + 0.5 * einsum('la,il->ia',  # N^5: O^3V^2 / N^4: O^2V^2
                       einsum('kd,klad->la', ul1, t2_1),
                       einsum('ijbc,jlbc->il', ur2, t2_1))
        - 1 * einsum('ic,ac->ia',
                     einsum('jb,ijbc->ic', ul1, t2_1),
                     einsum('ka,kc->ac', ur1, p0_2_ov))  # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum('jlac,ijlc->ia', ur2,  # N^6: O^4V^2 / N^4: O^2V^2
                     einsum('klcd,ijkd->ijlc', t2_1,
                            einsum('jb,ikbd->ijkd', ul1, t2_1)))
        - 1 * einsum('jkad,ijkd->ia', t2_1,  # N^6: O^4V^2 / N^4: O^2V^2
                     einsum('klcd,ijlc->ijkd', t2_1,
                            einsum('jb,ilbc->ijlc', ul1, ur2)))
        - 1 * einsum('ilad,ld->ia', t2_1,  # N^4: O^2V^2 / N^4: O^2V^2
                     einsum('klcd,kc->ld', t2_1, einsum('jb,jkbc->kc', ul1, ur2)))
        - 1 * einsum('ijkacd,jkcd->ia', t3_2,  # N^6: O^3V^3 / N^6: O^3V^3
                     einsum('jd,kc->jkcd', ul1, einsum('lb,klbc->kc', ur1, t2_1)))
        - 0.5 * einsum('ijkabd,jkbd->ia', t3_2,  # N^6: O^3V^3 / N^6: O^3V^3
                       einsum('klbd,jl->jkbd', t2_1, einsum('jc,lc->jl', ul1, ur1)))
        - 0.25 * einsum('jkla,ijkl->ia',  # N^6: O^4V^2 / N^4: O^2V^2
                        einsum('jb,klab->jkla', ul1, ur2),
                        einsum('ijcd,klcd->ijkl', t2_1, t2_1))
        - 0.25 * einsum('jkla,ijkl->ia',  # N^6: O^4V^2 / N^4: O^2V^2
                        einsum('jb,klab->jkla', ul1, t2_1),
                        einsum('ijcd,klcd->ijkl', ur2, t2_1))
    )
    dm.vo = (
        - 2 * einsum('ijab,jb->ai', ul2, ur1)  # N^4: O^2V^2 / N^4: O^2V^2
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1 * einsum('ka,ik->ai', ul1, einsum('jkbc,ijbc->ik', ur2, t2_1))
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1 * einsum('ka,ik->ai', ul1, einsum('jkbc,ijbc->ik', ur2, t2_2))
        # N^5: O^2V^3 / N^4: O^2V^2
        + 1 * einsum('ic,ac->ai', ul1, einsum('jkbc,jkab->ac', ur2, t2_2))
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum('ab,ib->ai', p0_2_vv, einsum('ijbc,jc->ib', ul2, ur1))
        # N^3: O^2V^1 / N^2: O^1V^1
        - 1 * einsum('ja,ij->ai', ul1, einsum('jb,ib->ij', ur1, p0_ov))
        # N^3: O^2V^1 / N^2: O^1V^1
        - 1 * einsum('ja,ij->ai', p0_ov, einsum('ib,jb->ij', ul1, ur1))
        # N^5: O^2V^3 / N^4: O^2V^2
        - 1 * einsum('ib,ab->ai', ul1, einsum('jkbc,jkac->ab', ur2, t2_1))
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum('ij,ja->ai', p0_2_oo, einsum('jkab,kb->ja', ul2, ur1))
        # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * einsum('ijac,jc->ai', t2_1, einsum('kb,jkbc->jc', ul1, ur2))
        # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * einsum('ikab,kb->ai', t2_2, einsum('jc,jkbc->kb', ul1, ur2))
        + 1 * einsum('iklacd,klcd->ai', t3_2,  # N^6: O^3V^3 / N^6: O^3V^3
                     einsum('kd,lc->klcd', ur1, einsum('jb,jlbc->lc', ul1, t2_1)))
        + 1 * einsum('ij,ja->ai',
                     einsum('ic,jc->ij', ul1, p0_2_ov),
                     einsum('kb,jkab->ja', ur1, t2_1))  # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum('ikab,kb->ai', t2_1,  # N^4: O^2V^2 / N^4: O^2V^2
                     einsum('jb,jk->kb', p0_2_ov, einsum('jc,kc->jk', ul1, ur1)))
        + 1 * einsum('ijab,jb->ai', t2_1,  # N^4: O^2V^2 / N^4: O^2V^2
                     einsum('jc,bc->jb', p0_2_ov, einsum('kc,kb->bc', ul1, ur1)))
        + 0.5 * einsum('ib,ab->ai', ul1,  # N^6: O^3V^3 / N^6: O^3V^3
                       einsum('klbc,klac->ab', t2_1,
                              einsum('jd,jklacd->klac', ur1, t3_2)))
        + 0.5 * einsum('iklabd,klbd->ai', t3_2,  # N^6: O^3V^3 / N^6: O^3V^3
                       einsum('klbc,cd->klbd', t2_1, einsum('jc,jd->cd', ul1, ur1)))
        + 0.5 * einsum('ijkabc,jkbc->ai', t3_2,  # N^6: O^3V^3 / N^6: O^3V^3
                       einsum('jlbc,kl->jkbc', t2_1, einsum('ld,kd->kl', ul1, ur1)))
        + 0.5 * einsum('ad,id->ai',  # N^5: O^2V^3 / N^4: O^2V^2
                       einsum('jlab,jlbd->ad', ul2, t2_1),
                       einsum('kc,ikcd->id', ur1, t2_1))
        + 0.5 * einsum('il,la->ai',  # N^5: O^3V^2 / N^4: O^2V^2
                       einsum('ijbd,jlbd->il', ul2, t2_1),
                       einsum('kc,klac->la', ur1, t2_1))
        + 0.5 * einsum('ikac,kc->ai', t2_1,  # N^5: O^2V^3 / N^4: O^2V^2
                       einsum('kb,bc->kc', ur1, einsum('jlbd,jlcd->bc', ul2, t2_1)))
        + 0.5 * einsum('ikac,kc->ai', t2_1,  # N^5: O^3V^2 / N^4: O^2V^2
                       einsum('jc,jk->kc', ur1, einsum('jlbd,klbd->jk', ul2, t2_1)))
        - 1 * einsum('ac,ic->ai',
                     einsum('ka,kc->ac', ul1, p0_2_ov),
                     einsum('jb,ijbc->ic', ur1, t2_1))  # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum('jlab,ijlb->ai', ul2,  # N^6: O^4V^2 / N^4: O^2V^2
                     einsum('klbd,ijkd->ijlb', t2_1,
                            einsum('jc,ikcd->ijkd', ur1, t2_1)))
        - 1 * einsum('klac,iklc->ai', t2_1,  # N^6: O^4V^2 / N^4: O^2V^2
                     einsum('jlbc,ijkb->iklc', t2_1,
                            einsum('ijbd,kd->ijkb', ul2, ur1)))
        - 1 * einsum('ikac,kc->ai', t2_1,  # N^4: O^2V^2 / N^4: O^2V^2
                     einsum('klbc,lb->kc', t2_1, einsum('jlbd,jd->lb', ul2, ur1)))
        - 0.5 * einsum('ja,ij->ai', ul1,  # N^6: O^3V^3 / N^6: O^3V^3
                       einsum('jlcd,ilcd->ij', t2_1,
                              einsum('kb,iklbcd->ilcd', ur1, t3_2)))
        - 0.25 * einsum('jkla,ijkl->ai',  # N^6: O^4V^2 / N^4: O^2V^2
                        einsum('klab,jb->jkla', ul2, ur1),
                        einsum('ijcd,klcd->ijkl', t2_1, t2_1))
        - 0.25 * einsum('ijkl,jkla->ai',  # N^6: O^4V^2 / N^4: O^2V^2
                        einsum('ijbd,klbd->ijkl', ul2, t2_1),
                        einsum('jc,klac->jkla', ur1, t2_1))
    )
    dm.vv = (
        + 1 * einsum('ia,ib->ab', ul1, ur1)  # N^3: O^1V^2 / N^2: V^2
        + 2 * einsum('ijac,ijbc->ab', ul2, ur2)  # N^5: O^2V^3 / N^4: O^2V^2
        # N^4: O^2V^2 / N^4: O^2V^2
        + 2 * einsum('ja,jb->ab', p0_2_ov, einsum('ic,ijbc->jb', ul1, ur2))
        # N^4: O^2V^2 / N^4: O^2V^2
        + 2 * einsum('jb,ja->ab', p0_2_ov, einsum('ijac,ic->ja', ul2, ur1))
        # N^3: O^1V^2 / N^2: V^2
        - 0.5 * einsum('ia,ib->ab', ul1, einsum('ic,bc->ib', ur1, p0_vv))
        # N^3: O^1V^2 / N^2: V^2
        - 0.5 * einsum('ib,ia->ab', ur1, einsum('ic,ac->ia', ul1, p0_vv))
        + 1 * einsum('kb,ka->ab',
                     einsum('ic,ikbc->kb', ul1, t2_1),
                     einsum('jd,jkad->ka', ur1, t2_1))  # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum('jkac,jkbc->ab', t2_2,  # N^5: O^2V^3 / N^4: O^2V^2
                     einsum('ijbc,ik->jkbc', t2_1, einsum('id,kd->ik', ul1, ur1)))
        + 0.5 * einsum('ia,ib->ab', ul1,  # N^4: O^2V^2 / N^4: O^2V^2
                       einsum('ikbd,kd->ib', t2_1,
                              einsum('jc,jkcd->kd', ur1, t2_1)))
        + 0.5 * einsum('ja,jb->ab', ul1,  # N^4: O^2V^2 / N^4: O^2V^2
                       einsum('jkbd,kd->jb', t2_1,
                              einsum('ic,ikcd->kd', ur1, t2_2)))
        + 0.5 * einsum('ja,jb->ab', ul1,  # N^4: O^2V^2 / N^4: O^2V^2
                       einsum('jkbd,kd->jb', t2_2,
                              einsum('ic,ikcd->kd', ur1, t2_1)))
        - 1 * einsum('jb,ja->ab',
                     einsum('id,ijbd->jb', ul1, t2_2),
                     einsum('kc,jkac->ja', ur1, t2_1))  # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum('jb,ja->ab',
                     einsum('id,ijbd->jb', ul1, t2_1),
                     einsum('kc,jkac->ja', ur1, t2_2))  # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum('ikbc,ikac->ab', t2_1,  # N^5: O^2V^3 / N^4: O^2V^2
                     einsum('jkac,ij->ikac', t2_1, einsum('id,jd->ij', ul1, ur1)))
        - 1 * einsum('ijbc,ijac->ab', t2_2,  # N^5: O^2V^3 / N^4: O^2V^2
                     einsum('ikac,jk->ijac', t2_1, einsum('jd,kd->jk', ul1, ur1)))
        - 0.5 * einsum('kb,ka->ab', ur1,  # N^4: O^2V^2 / N^4: O^2V^2
                       einsum('jkad,jd->ka', t2_1,
                              einsum('ic,ijcd->jd', ul1, t2_2)))
        - 0.5 * einsum('kb,ka->ab', ur1,  # N^4: O^2V^2 / N^4: O^2V^2
                       einsum('jkad,jd->ka', t2_2,
                              einsum('ic,ijcd->jd', ul1, t2_1)))
        - 0.5 * einsum('jb,ja->ab', ur1,  # N^4: O^2V^2 / N^4: O^2V^2
                       einsum('jkac,kc->ja', t2_1,
                              einsum('id,ikcd->kc', ul1, t2_1)))
        - 0.5 * einsum('jkbd,jkad->ab', t2_1,  # N^5: O^2V^3 / N^4: O^2V^2
                       einsum('jkac,cd->jkad', t2_1, einsum('id,ic->cd', ul1, ur1)))
        - 0.5 * einsum('jkbd,jkad->ab', t2_2,  # N^5: O^2V^3 / N^4: O^2V^2
                       einsum('jkac,cd->jkad', t2_1, einsum('id,ic->cd', ul1, ur1)))
        - 0.5 * einsum('jkac,jkbc->ab', t2_2,  # N^5: O^2V^3 / N^4: O^2V^2
                       einsum('jkbd,cd->jkbc', t2_1, einsum('id,ic->cd', ul1, ur1)))
    )
    return dm


# Ref: https://doi.org/10.1080/00268976.2013.859313
DISPATCH = {"adc0": s2s_tdm_adc0,
            "adc1": s2s_tdm_adc0,       # same as ADC(0)
            "adc2": s2s_tdm_adc2,
            "adc2x": s2s_tdm_adc2,      # same as ADC(2)
            "adc3": s2s_tdm_adc3,
            }


def state2state_transition_dm(method, ground_state, amplitude_from,
                              amplitude_to, intermediates=None):
    """
    Compute the state to state transition density matrix
    state in the MO basis using the intermediate-states representation.

    Parameters
    ----------
    method : str, AdcMethod
        The method to use for the computation (e.g. "adc2")
    ground_state : LazyMp
        The ground state upon which the excitation was based
    amplitude_from : AmplitudeVector
        The amplitude vector of the state to start from
    amplitude_to : AmplitudeVector
        The amplitude vector of the state to excite to
    intermediates : adcc.Intermediates
        Intermediates from the ADC calculation to reuse
    """
    if not isinstance(method, AdcMethod):
        method = AdcMethod(method)
    if not isinstance(ground_state, LazyMp):
        raise TypeError("ground_state should be a LazyMp object.")
    if not isinstance(amplitude_from, AmplitudeVector):
        raise TypeError("amplitude_from should be an AmplitudeVector object.")
    if not isinstance(amplitude_to, AmplitudeVector):
        raise TypeError("amplitude_to should be an AmplitudeVector object.")
    if intermediates is None:
        intermediates = Intermediates(ground_state)

    if method.name not in DISPATCH:
        raise NotImplementedError("state2state_transition_dm is not implemented "
                                  f"for {method.name}.")
    else:
        # final state is on the bra side/left (complex conjugate)
        # see ref https://doi.org/10.1080/00268976.2013.859313, appendix A2
        ret = DISPATCH[method.name](ground_state, amplitude_to, amplitude_from,
                                    intermediates)
        return ret.evaluate()
