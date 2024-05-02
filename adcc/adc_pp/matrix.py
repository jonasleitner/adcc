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
from collections import namedtuple

from adcc import block as b
from adcc.functions import direct_sum, einsum, zeros_like
from adcc.Intermediates import Intermediates, register_as_intermediate
from adcc.AmplitudeVector import AmplitudeVector

__all__ = ["block"]

# TODO One thing one could still do to improve timings is implement a "fast einsum"
#      that does not call opt_einsum, but directly dispatches to libadcc. This could
#      lower the call overhead in the applies for the cases where we have only a
#      trivial einsum to do. For the moment I'm not convinced that is worth the
#      effort ... I suppose it only makes a difference for the cheaper ADC variants
#      (ADC(0), ADC(1), CVS-ADC(0-2)-x), but then on the other hand they are not
#      really so much our focus.


#
# Dispatch routine
#
"""
`apply` is a function mapping an AmplitudeVector to the contribution of this
block to the result of applying the ADC matrix. `diagonal` is an `AmplitudeVector`
containing the expression to the diagonal of the ADC matrix from this block.
"""
AdcBlock = namedtuple("AdcBlock", ["apply", "diagonal"])


def block(ground_state, spaces, order, variant=None, intermediates=None):
    """
    Gets ground state, potentially intermediates, spaces (ph, pphh and so on)
    and the perturbation theory order for the block,
    variant is "cvs" or sth like that.

    It is assumed largely, that CVS is equivalent to mp.has_core_occupied_space,
    while one would probably want in the long run that one can have an "o2" space,
    but not do CVS.
    """
    if isinstance(variant, str):
        variant = [variant]
    elif variant is None:
        variant = []
    reference_state = ground_state.reference_state
    if intermediates is None:
        intermediates = Intermediates(ground_state)

    if ground_state.has_core_occupied_space and "cvs" not in variant:
        raise ValueError("Cannot run a general (non-core-valence approximated) "
                         "ADC method on top of a ground state with a "
                         "core-valence separation.")
    if not ground_state.has_core_occupied_space and "cvs" in variant:
        raise ValueError("Cannot run a core-valence approximated ADC method on "
                         "top of a ground state without a "
                         "core-valence separation.")
    if "re" in variant and "cvs" in variant:
        raise NotImplementedError("Core-valence-approximated RE-ADC not "
                                  "implemented.")

    fn = "_".join(["block"] + variant + spaces + [str(order)])

    if fn not in globals():
        raise ValueError("Could not dispatch: "
                         f"spaces={spaces} order={order} variant={variant}")
    return globals()[fn](reference_state, ground_state, intermediates)


#
# 0th order main
#
def block_ph_ph_0(hf, mp, intermediates):
    fCC = hf.fcc if hf.has_core_occupied_space else hf.foo
    diagonal = AmplitudeVector(ph=direct_sum("a-i->ia", hf.fvv.diagonal(),
                                             fCC.diagonal()))

    def apply(ampl):
        return AmplitudeVector(ph=(
            + einsum("ib,ab->ia", ampl.ph, hf.fvv)
            - einsum("IJ,Ja->Ia", fCC, ampl.ph)
        ))
    return AdcBlock(apply, diagonal)


block_cvs_ph_ph_0 = block_ph_ph_0


def block_re_ph_ph_0(hf, re, intermediates):
    # is identical to ADC(1)
    return block_ph_ph_1(hf, re, intermediates)


def diagonal_pphh_pphh_0(hf):
    # Note: adcman similarly does not symmetrise the occupied indices
    #       (for both CVS and general ADC)
    fCC = hf.fcc if hf.has_core_occupied_space else hf.foo
    res = direct_sum("-i-J+a+b->iJab",
                     hf.foo.diagonal(), fCC.diagonal(),
                     hf.fvv.diagonal(), hf.fvv.diagonal())
    return AmplitudeVector(pphh=res.symmetrise(2, 3))


def block_pphh_pphh_0(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            + 2 * einsum("ijac,bc->ijab", ampl.pphh, hf.fvv).antisymmetrise(2, 3)
            - 2 * einsum("ik,kjab->ijab", hf.foo, ampl.pphh).antisymmetrise(0, 1)
        ))
    return AdcBlock(apply, diagonal_pphh_pphh_0(hf))


def block_cvs_pphh_pphh_0(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            + 2 * einsum("iJac,bc->iJab", ampl.pphh, hf.fvv).antisymmetrise(2, 3)
            - einsum("ik,kJab->iJab", hf.foo, ampl.pphh)
            - einsum("JK,iKab->iJab", hf.fcc, ampl.pphh)
        ))
    return AdcBlock(apply, diagonal_pphh_pphh_0(hf))


def block_re_pphh_pphh_0(hf, re, intermediates):
    return block_pphh_pphh_1(hf, re, intermediates)


#
# 0th order coupling
#
def block_ph_pphh_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


def block_pphh_ph_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


block_cvs_ph_pphh_0 = block_ph_pphh_0
block_cvs_pphh_ph_0 = block_pphh_ph_0

block_re_ph_pphh_0 = block_ph_pphh_0
block_re_pphh_ph_0 = block_pphh_ph_0


#
# 1st order main
#
def block_ph_ph_1(hf, mp, intermediates):
    fCC = hf.fcc if hf.has_core_occupied_space else hf.foo
    CvCv = hf.cvcv if hf.has_core_occupied_space else hf.ovov
    diagonal = AmplitudeVector(ph=(
        + direct_sum("a-i->ia", hf.fvv.diagonal(), fCC.diagonal())  # order 0
        - einsum("IaIa->Ia", CvCv)  # order 1
    ))

    def apply(ampl):
        return AmplitudeVector(ph=(                 # PT order
            + einsum("ib,ab->ia", ampl.ph, hf.fvv)  # 0
            - einsum("IJ,Ja->Ia", fCC, ampl.ph)     # 0
            - einsum("JaIb,Jb->Ia", CvCv, ampl.ph)  # 1
        ))
    return AdcBlock(apply, diagonal)


block_cvs_ph_ph_1 = block_ph_ph_1

# no first order contribution for RE-ADC
block_re_ph_ph_1 = block_re_ph_ph_0


def diagonal_pphh_pphh_1(hf):
    # Fock matrix and ovov diagonal term (sometimes called "intermediate diagonal")
    dinterm_ov = (direct_sum("a-i->ia", hf.fvv.diagonal(), hf.foo.diagonal())
                  - 2.0 * einsum("iaia->ia", hf.ovov)).evaluate()

    if hf.has_core_occupied_space:
        dinterm_Cv = (direct_sum("a-I->Ia", hf.fvv.diagonal(), hf.fcc.diagonal())
                      - 2.0 * einsum("IaIa->Ia", hf.cvcv)).evaluate()
        diag_oC = einsum("iJiJ->iJ", hf.ococ)
    else:
        dinterm_Cv = dinterm_ov
        diag_oC = einsum("ijij->ij", hf.oooo).symmetrise()

    diag_vv = einsum("abab->ab", hf.vvvv).symmetrise()
    return AmplitudeVector(pphh=(
        + direct_sum("ia+Jb->iJab", dinterm_ov, dinterm_Cv).symmetrise(2, 3)
        + direct_sum("iJ+ab->iJab", diag_oC, diag_vv)
    ))


def block_pphh_pphh_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(  # 0th order
            + 2 * einsum("ijac,bc->ijab", ampl.pphh, hf.fvv).antisymmetrise(2, 3)
            - 2 * einsum("ik,kjab->ijab", hf.foo, ampl.pphh).antisymmetrise(0, 1)
            # 1st order
            + (
                -4 * einsum("ikac,kbjc->ijab", ampl.pphh, hf.ovov)
            ).antisymmetrise(0, 1).antisymmetrise(2, 3)
            + 0.5 * einsum("ijkl,klab->ijab", hf.oooo, ampl.pphh)
            + 0.5 * einsum("ijcd,abcd->ijab", ampl.pphh, hf.vvvv)
        ))
    return AdcBlock(apply, diagonal_pphh_pphh_1(hf))


def block_cvs_pphh_pphh_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            # 0th order
            + 2.0 * einsum("iJac,bc->iJab", ampl.pphh, hf.fvv).antisymmetrise(2, 3)
            - 1.0 * einsum("ik,kJab->iJab", hf.foo, ampl.pphh)
            - 1.0 * einsum("JK,iKab->iJab", hf.fcc, ampl.pphh)
            # 1st order
            + (
                - 2.0 * einsum("iKac,KbJc->iJab", ampl.pphh, hf.cvcv)
                + 2.0 * einsum("icka,kJbc->iJab", hf.ovov, ampl.pphh)
            ).antisymmetrise(2, 3)
            + 1.0 * einsum("iJlK,lKab->iJab", hf.ococ, ampl.pphh)
            + 0.5 * einsum("iJcd,abcd->iJab", ampl.pphh, hf.vvvv)
        ))
    return AdcBlock(apply, diagonal_pphh_pphh_1(hf))


block_re_pphh_pphh_1 = block_re_pphh_pphh_0


#
# 1st order coupling
#
def block_ph_pphh_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph=(
            + einsum("jkib,jkab->ia", hf.ooov, ampl.pphh)
            + einsum("ijbc,jabc->ia", ampl.pphh, hf.ovvv)
        ))
    return AdcBlock(apply, 0)


def block_cvs_ph_pphh_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph=(
            + sqrt(2) * einsum("jKIb,jKab->Ia", hf.occv, ampl.pphh)
            - 1 / sqrt(2) * einsum("jIbc,jabc->Ia", ampl.pphh, hf.ovvv)
        ))
    return AdcBlock(apply, 0)


block_re_ph_pphh_1 = block_ph_pphh_1


def block_pphh_ph_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            + einsum("ic,jcab->ijab", ampl.ph, hf.ovvv).antisymmetrise(0, 1)
            - einsum("ijka,kb->ijab", hf.ooov, ampl.ph).antisymmetrise(2, 3)
        ))
    return AdcBlock(apply, 0)


def block_cvs_pphh_ph_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            + sqrt(2) * einsum("jIKb,Ka->jIab",
                               hf.occv, ampl.ph).antisymmetrise(2, 3)
            - 1 / sqrt(2) * einsum("Ic,jcab->jIab", ampl.ph, hf.ovvv)
        ))
    return AdcBlock(apply, 0)


block_re_pphh_ph_1 = block_pphh_ph_1


#
# 2nd order main
#
def block_ph_ph_2(hf, mp, intermediates):
    i1 = intermediates.adc2_i1
    i2 = intermediates.adc2_i2
    diagonal = AmplitudeVector(ph=(
        + direct_sum("a-i->ia", i1.diagonal(), i2.diagonal())
        - einsum("IaIa->Ia", hf.ovov)
        - einsum("ikac,ikac->ia", mp.t2oo, hf.oovv)
    ))

    # Not used anywhere else, so kept as an anonymous intermediate
    term_t2_eri = (
        + einsum("ijab,jkbc->ikac", mp.t2oo, hf.oovv)
        + einsum("ijab,jkbc->ikac", hf.oovv, mp.t2oo)
    ).evaluate()

    def apply(ampl):
        return AmplitudeVector(ph=(
            + einsum("ib,ab->ia", ampl.ph, i1)
            - einsum("ij,ja->ia", i2, ampl.ph)
            - einsum("jaib,jb->ia", hf.ovov, ampl.ph)    # 1
            - 0.5 * einsum("ikac,kc->ia", term_t2_eri, ampl.ph)  # 2
        ))
    return AdcBlock(apply, diagonal)


def block_cvs_ph_ph_2(hf, mp, intermediates):
    i1 = intermediates.adc2_i1
    diagonal = AmplitudeVector(ph=(
        + direct_sum("a-i->ia", i1.diagonal(), hf.fcc.diagonal())
        - einsum("IaIa->Ia", hf.cvcv)
    ))

    def apply(ampl):
        return AmplitudeVector(ph=(
            + einsum("ib,ab->ia", ampl.ph, i1)
            - einsum("ij,ja->ia", hf.fcc, ampl.ph)
            - einsum("JaIb,Jb->Ia", hf.cvcv, ampl.ph)
        ))
    return AdcBlock(apply, diagonal)


def block_re_ph_ph_2(hf, re, intermediates):
    m11 = intermediates.re_adc2_m11  # is already evaluated in __getattr__
    diagonal = AmplitudeVector(ph=einsum('iaia->ia', m11))

    def apply(ampl):
        return AmplitudeVector(ph=einsum('iajb,jb->ia', m11, ampl.ph))
    return AdcBlock(apply, diagonal)


#
# 2nd order coupling
#
def block_ph_pphh_2(hf, mp, intermediates):
    pia_ooov = intermediates.adc3_pia
    pib_ovvv = intermediates.adc3_pib

    def apply(ampl):
        return AmplitudeVector(ph=(
            + einsum("jkib,jkab->ia", pia_ooov, ampl.pphh)
            + einsum("ijbc,jabc->ia", ampl.pphh, pib_ovvv)
            + einsum("icab,jkcd,jkbd->ia", hf.ovvv, ampl.pphh, mp.t2oo)  # 2nd
            + einsum("ijka,jlbc,klbc->ia", hf.ooov, mp.t2oo, ampl.pphh)  # 2nd
        ))
    return AdcBlock(apply, 0)


def block_cvs_ph_pphh_2(hf, mp, intermediates):
    pia_occv = intermediates.cvs_adc3_pia
    pib_ovvv = intermediates.adc3_pib

    def apply(ampl):
        return AmplitudeVector(ph=(1 / sqrt(2)) * (
            + 2.0 * einsum("lKIc,lKac->Ia", pia_occv, ampl.pphh)
            - einsum("lIcd,lacd->Ia", ampl.pphh, pib_ovvv)
            - einsum("jIKa,ljcd,lKcd->Ia", hf.occv, mp.t2oo, ampl.pphh)
        ))
    return AdcBlock(apply, 0)


def block_re_ph_pphh_2(hf, re, intermediates):
    t2_1 = re.t2(b.oovv)
    t1_2 = re.ts2(b.ov)

    t2eri_A = intermediates.adc3_pia  # also includes the first order term
    t2eri_B = intermediates.adc3_pib  # also includes the first order term

    def apply(ampl):
        ur2 = ampl.pphh
        # The scaling comment is given as: [comp_scaling] / [mem_scaling]
        return AmplitudeVector(ph=(
            # 2nd order
            + 1 * einsum('icad,cd->ia', hf.ovvv,  # N^5: O^2V^3 / N^4: O^1V^3
                         einsum('jkbc,jkbd->cd', ur2, t2_1))
            + 1 * einsum('ilka,kl->ia', hf.ooov,  # N^5: O^3V^2 / N^4: O^2V^2
                         einsum('jkbc,jlbc->kl', ur2, t2_1))
            + 1 * einsum('ijbc,jabc->ia', ur2, t2eri_B)  # N^5: O^2V^3 / N^4: O^1V^3
            - 1 * einsum('adbc,ibcd->ia', hf.vvvv,  # N^5: O^1V^4 / N^4: V^4
                         einsum('ijbc,jd->ibcd', ur2, t1_2))
            + 2 * einsum('jakb,ijkb->ia', hf.ovov,  # N^5: O^3V^2 / N^4: O^2V^2
                         einsum('ijbc,kc->ijkb', ur2, t1_2))
            - 1 * einsum('ijab,jb->ia', ur2,  # N^5: O^2V^3 / N^4: O^1V^3
                         einsum('kbcd,jkcd->jb', hf.ovvv, t2_1))
            - 1 * einsum('ijab,jb->ia', ur2,  # N^5: O^3V^2 / N^4: O^2V^2
                         einsum('kljc,klbc->jb', hf.ooov, t2_1))
            - 2 * einsum('ijab,jb->ia', ur2,  # N^4: O^2V^2 / N^4: O^2V^2
                         einsum('bc,jc->jb', hf.fvv, t1_2))
            + 2 * einsum('ijab,jb->ia', ur2,  # N^4: O^2V^2 / N^4: O^2V^2
                         einsum('jckb,kc->jb', hf.ovov, t1_2))
            + 2 * einsum('ijab,jb->ia', ur2,  # N^4: O^2V^2 / N^4: O^2V^2
                         einsum('jk,kb->jb', hf.foo, t1_2))
            + 1 * einsum('jkab,jkib->ia', ur2, t2eri_A)  # N^5: O^3V^2 / N^4: O^2V^2
            - 1 * einsum('jkab,ijkb->ia', ur2,  # N^5: O^3V^2 / N^4: O^2V^2
                         einsum('iljk,lb->ijkb', hf.oooo, t1_2))
            + 2 * einsum('jkab,ijkb->ia', ur2,  # N^5: O^3V^2 / N^4: O^2V^2
                         einsum('ibjc,kc->ijkb', hf.ovov, t1_2))
        ))
    return AdcBlock(apply, 0)


def block_pphh_ph_2(hf, mp, intermediates):
    pia_ooov = intermediates.adc3_pia
    pib_ovvv = intermediates.adc3_pib

    def apply(ampl):
        return AmplitudeVector(pphh=(
            (
                + einsum("ic,jcab->ijab", ampl.ph, pib_ovvv)
                + einsum("lkic,kc,jlab->ijab", hf.ooov, ampl.ph, mp.t2oo)  # 2st
            ).antisymmetrise(0, 1)
            + (
                - einsum("ijka,kb->ijab", pia_ooov, ampl.ph)
                - einsum("ijac,kbcd,kd->ijab", mp.t2oo, hf.ovvv, ampl.ph)  # 2st
            ).antisymmetrise(2, 3)
        ))
    return AdcBlock(apply, 0)


def block_cvs_pphh_ph_2(hf, mp, intermediates):
    pia_occv = intermediates.cvs_adc3_pia
    pib_ovvv = intermediates.adc3_pib

    def apply(ampl):
        return AmplitudeVector(pphh=(1 / sqrt(2)) * (
            - 2.0 * einsum("jIKa,Kb->jIab", pia_occv, ampl.ph).antisymmetrise(2, 3)
            - einsum("Ic,jcab->jIab", ampl.ph, pib_ovvv)
            - einsum("lKIc,Kc,jlab->jIab", hf.occv, ampl.ph, mp.t2oo)
        ))
    return AdcBlock(apply, 0)


def block_re_pphh_ph_2(hf, re, intermediates):
    t2_1 = re.t2(b.oovv)
    t1_2 = re.ts2(b.ov)

    t2eri_A = intermediates.adc3_pia  # also includes first order term
    t2eri_B = intermediates.adc3_pib  # also includes first order term

    def apply(ampl):
        ur1 = ampl.ph
        # The scaling comment is given as: [comp_scaling] / [mem_scaling]
        return AmplitudeVector(pphh=(
            # 2nd order
            + 2 * (
                + 0.5 * einsum('ijad,bd->ijab', t2_1,  # N^5: O^2V^3 / N^4: O^1V^3
                               einsum('kbcd,kc->bd', hf.ovvv, ur1))
                + 0.5 * einsum('ilab,jl->ijab', t2_1,  # N^5: O^3V^2 / N^4: O^2V^2
                               einsum('kljc,kc->jl', hf.ooov, ur1))
                # N^5: O^2V^3 / N^4: O^1V^3
                + 0.5 * einsum('ic,jcab->ijab', ur1, t2eri_B)
                + 0.5 * einsum('id,jabd->ijab', t1_2,  # N^5: O^1V^4 / N^4: V^4
                               einsum('abcd,jc->jabd', hf.vvvv, ur1))
                # N^5: O^3V^2 / N^4: O^2V^2
                - 0.5 * einsum('kb,ijka->ijab', ur1, t2eri_A)
                + 0.5 * einsum('la,ijlb->ijab', t1_2,  # N^5: O^3V^2 / N^4: O^2V^2
                               einsum('ijkl,kb->ijlb', hf.oooo, ur1))
            )
            + 4 * (
                + 0.5 * einsum('ka,ijkb->ijab', t1_2,  # N^5: O^3V^2 / N^4: O^2V^2
                               einsum('ickb,jc->ijkb', hf.ovov, ur1))
                + 0.5 * einsum('jb,ia->ijab', ur1,  # N^4: O^2V^2 / N^4: O^2V^2
                               einsum('icka,kc->ia', hf.ovov, t1_2))
                + 0.5 * einsum('ia,jb->ijab', ur1,  # N^4: O^2V^2 / N^4: O^2V^2
                               einsum('jk,kb->jb', hf.foo, t1_2))
                + 0.5 * einsum('ja,ib->ijab', ur1,  # N^4: O^2V^2 / N^4: O^2V^2
                               einsum('bc,ic->ib', hf.fvv, t1_2))
                - 0.25 * einsum('jb,ia->ijab', ur1,  # N^5: O^2V^3 / N^4: O^1V^3
                                einsum('kacd,ikcd->ia', hf.ovvv, t2_1))
                - 0.25 * einsum('jb,ia->ijab', ur1,  # N^5: O^3V^2 / N^4: O^2V^2
                                einsum('klic,klac->ia', hf.ooov, t2_1))
                + 0.5 * einsum('ka,ijkb->ijab', ur1,  # N^5: O^3V^2 / N^4: O^2V^2
                               einsum('ickb,jc->ijkb', hf.ovov, t1_2))
            )
        ).antisymmetrise(0, 1).antisymmetrise(2, 3))
    return AdcBlock(apply, 0)


#
# 3rd order main
#
def block_ph_ph_3(hf, mp, intermediates):
    if hf.has_core_occupied_space:
        m11 = intermediates.cvs_adc3_m11
    else:
        m11 = intermediates.adc3_m11
    diagonal = AmplitudeVector(ph=einsum("iaia->ia", m11))

    def apply(ampl):
        return AmplitudeVector(ph=einsum("iajb,jb->ia", m11, ampl.ph))
    return AdcBlock(apply, diagonal)


block_cvs_ph_ph_3 = block_ph_ph_3


def block_re_ph_ph_3(hf, re, intermediates):
    m11 = intermediates.re_adc3_m11

    diagonal = AmplitudeVector(ph=einsum("iaia->ia", m11))

    def apply(ampl):
        return AmplitudeVector(ph=einsum("iajb,jb->ia", m11, ampl.ph))
    return AdcBlock(apply, diagonal)


#
# Intermediates
#

@register_as_intermediate
def adc2_i1(hf, mp, intermediates):
    # This definition differs from libadc. It additionally has the hf.fvv term.
    return hf.fvv + 0.5 * einsum("ijac,ijbc->ab", mp.t2oo, hf.oovv).symmetrise()


@register_as_intermediate
def adc2_i2(hf, mp, intermediates):
    # This definition differs from libadc. It additionally has the hf.foo term.
    return hf.foo - 0.5 * einsum("ikab,jkab->ij", mp.t2oo, hf.oovv).symmetrise()


@register_as_intermediate
def re_adc2_m11(hf, re, intermediates):
    t2_1 = re.t2(b.oovv)

    p0 = re.diffdm(2)
    p0_2_oo = p0.oo
    p0_2_vv = p0.vv

    t2eri_3 = re.t2eri(b.oovv, b.oo)
    t2eri_4 = re.t2eri(b.oovv, b.ov)
    t2eri_5 = re.t2eri(b.oovv, b.vv)

    t2sq = einsum("ikac,jkbc->iajb", t2_1, t2_1).evaluate()

    # Build two Kronecker deltas
    d_oo = zeros_like(hf.foo)
    d_vv = zeros_like(hf.fvv)
    d_oo.set_mask("ii", 1.0)
    d_vv.set_mask("aa", 1.0)

    return (
        # The scaling comment is given as: [comp_scaling] / [mem_scaling]
        # 0th order contributions:
        - 1 * einsum('ij,ab->iajb', hf.foo, d_vv)  # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum('ab,ij->iajb', hf.fvv, d_oo)  # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum('ibja->iajb', hf.ovov)  # N^4: O^2V^2 / N^4: O^2V^2
        # 2nd order contributions:
        + 2 * (  # terms with (1 + P_ab P_ij)
            # N^5: O^2V^3 / N^4: O^2V^2
            + 0.5 * einsum('ibjc,ac->iajb', hf.ovov, p0_2_vv)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 1 * einsum('ibkc,jcka->iajb', hf.ovov, t2sq)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.5 * einsum('ibka,jk->iajb', hf.ovov, p0_2_oo)
        )
        + 1 * einsum('ikac,jkbc->iajb', t2_1, t2eri_4)  # N^6: O^3V^3 / N^4: O^2V^2
        + 1 * einsum('ikac,kjcb->iajb', t2_1, t2eri_4)  # N^6: O^3V^3 / N^4: O^2V^2
        + 0.5 * einsum('bc,iajc->iajb', hf.fvv, t2sq)  # N^5: O^2V^3 / N^4: O^2V^2
        + 0.5 * einsum('ik,jbka->iajb', hf.foo, t2sq)  # N^5: O^3V^2 / N^4: O^2V^2
        # N^6: O^3V^3 / N^4: O^2V^2
        + 0.5 * einsum('ikac,jkbc->iajb', t2_1, t2eri_3)
        # N^6: O^3V^3 / N^4: O^2V^2
        + 0.5 * einsum('ikac,jkbc->iajb', t2_1, t2eri_5)
        # N^6: O^3V^3 / N^4: O^2V^2
        + 0.5 * einsum('jkbc,ikca->iajb', t2_1, t2eri_4)
        - 1 * einsum('adbc,icjd->iajb', hf.vvvv, t2sq)  # N^6: O^2V^4 / N^4: V^4
        - 1 * einsum('iljk,kalb->iajb', hf.oooo, t2sq)  # N^6: O^4V^2 / N^4: O^2V^2
        - 1 * einsum('jkbc,ikac->iajb', hf.oovv, t2_1)  # N^6: O^3V^3 / N^4: O^2V^2
        - 0.5 * einsum('ac,icjb->iajb', hf.fvv, t2sq)  # N^5: O^2V^3 / N^4: O^2V^2
        - 0.5 * einsum('jk,iakb->iajb', hf.foo, t2sq)  # N^5: O^3V^2 / N^4: O^2V^2
        # N^6: O^3V^3 / N^4: O^2V^2
        - 0.5 * einsum('ikac,jkcb->iajb', t2_1, t2eri_4)
        + 0.5 * einsum('icjd,abcd->iajb', hf.ovov,  # N^6: O^2V^4 / N^4: V^4
                       einsum('klac,klbd->abcd', t2_1, t2_1))
        + 0.5 * einsum('kalb,ijkl->iajb', hf.ovov,  # N^6: O^4V^2 / N^4: O^2V^2
                       einsum('ikcd,jlcd->ijkl', t2_1, t2_1))
        + 1 * einsum('ij,ab->iajb', d_oo,  # N^4: V^4 / N^4: V^4
                     einsum('adbc,cd->ab', hf.vvvv, p0_2_vv))
        + 1 * einsum('ij,ab->iajb', d_oo,  # N^4: O^2V^2 / N^4: O^2V^2
                     einsum('kalb,kl->ab', hf.ovov, p0_2_oo))
        + 1 * einsum('ij,ab->iajb', d_oo,  # N^5: O^2V^3 / N^4: O^2V^2
                     einsum('klac,klcb->ab', t2_1, t2eri_4))
        + 0.5 * einsum('ij,ab->iajb', d_oo,  # N^5: O^2V^3 / N^4: O^2V^2
                       einsum('klbc,klac->ab', hf.oovv, t2_1))
        + 0.5 * einsum('ij,ab->iajb', d_oo,  # N^4: O^2V^2 / N^4: O^2V^2
                       einsum('ac,bc->ab', hf.fvv, p0_2_vv))
        - 0.5 * einsum('ij,ab->iajb', d_oo,  # N^4: O^2V^2 / N^4: O^2V^2
                       einsum('bc,ac->ab', hf.fvv, p0_2_vv))
        - 0.25 * einsum('ij,ab->iajb', d_oo,  # N^5: O^2V^3 / N^4: O^2V^2
                        einsum('klac,klbc->ab', t2_1, t2eri_5))
        + 1 * einsum('ab,ij->iajb', d_vv,  # N^5: O^3V^2 / N^4: O^2V^2
                     einsum('ikcd,jkdc->ij', t2_1, t2eri_4))
        + 0.5 * einsum('ab,ij->iajb', d_vv,  # N^5: O^3V^2 / N^4: O^2V^2
                       einsum('jkcd,ikcd->ij', hf.oovv, t2_1))
        + 0.5 * einsum('ab,ij->iajb', d_vv,  # N^4: O^2V^2 / N^4: O^2V^2
                       einsum('ik,jk->ij', hf.foo, p0_2_oo))
        - 1 * einsum('ab,ij->iajb', d_vv,  # N^4: O^2V^2 / N^4: O^2V^2
                     einsum('icjd,cd->ij', hf.ovov, p0_2_vv))
        - 1 * einsum('ab,ij->iajb', d_vv,  # N^4: O^2V^2 / N^4: O^2V^2
                     einsum('iljk,kl->ij', hf.oooo, p0_2_oo))
        - 0.5 * einsum('ab,ij->iajb', d_vv,  # N^4: O^2V^2 / N^4: O^2V^2
                       einsum('jk,ik->ij', hf.foo, p0_2_oo))
        - 0.25 * einsum('ab,ij->iajb', d_vv,  # N^5: O^3V^2 / N^4: O^2V^2
                        einsum('ikcd,jkcd->ij', t2_1, t2eri_3))
    ).symmetrise((0, 2), (1, 3))


@register_as_intermediate
def re_adc3_m11(hf, re, intermediates):
    t1_2 = re.ts2(b.ov)

    # Build two Kronecker deltas
    d_oo = zeros_like(hf.foo)
    d_vv = zeros_like(hf.fvv)
    d_oo.set_mask("ii", 1.0)
    d_vv.set_mask("aa", 1.0)

    return (
        # evaluate the 0th, 1st and 2nd order contributions on the fly
        # avoids caching of the re_adc2_m11 intermediate, which should not be
        # required for a re-adc3 calculation
        re_adc2_m11(hf, re, intermediates)
        + 2 * (
            # N^5: O^2V^3 / N^4: O^1V^3
            + 1 * einsum('ibac,jc->iajb', hf.ovvv, t1_2)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum('ikja,kb->iajb', hf.ooov, t1_2)
            - 1 * einsum('ij,ab->iajb', d_oo,  # N^4: O^1V^3 / N^4: O^1V^3
                         einsum('kabc,kc->ab', hf.ovvv, t1_2))
            - 1 * einsum('ab,ij->iajb', d_vv,  # N^4: O^2V^2 / N^4: O^2V^2
                         einsum('ikjc,kc->ij', hf.ooov, t1_2))
        ).symmetrise((0, 2), (1, 3))
    )


def adc3_i1(hf, mp, intermediates):
    # Used for both CVS and general
    td2 = mp.td2(b.oovv)
    p0 = intermediates.cvs_p0 if hf.has_core_occupied_space else mp.diffdm(2)

    t2eri_sum = (
        + einsum("jicb->ijcb", mp.t2eri(b.oovv, b.ov))  # t2eri4
        - 0.25 * mp.t2eri(b.oovv, b.vv)                 # t2eri5
    )
    return (
        (  # symmetrise a<>b
            + 0.5 * einsum("ijac,ijbc->ab", mp.t2oo + td2, hf.oovv)
            - 1.0 * einsum("ijac,ijcb->ab", mp.t2oo, t2eri_sum)
            - 2.0 * einsum("iabc,ic->ab", hf.ovvv, p0.ov)
        ).symmetrise()
        + einsum("iajb,ij->ab", hf.ovov, p0.oo)
        + einsum("acbd,cd->ab", hf.vvvv, p0.vv)
    )


def adc3_i2(hf, mp, intermediates):
    # Used only for general
    td2 = mp.td2(b.oovv)
    p0 = mp.diffdm(2)

    # t2eri4 + t2eri3 / 4
    t2eri_sum = mp.t2eri(b.oovv, b.ov) + 0.25 * mp.t2eri(b.oovv, b.oo)
    return (
        (  # symmetrise i<>j
            + 0.5 * einsum("ikab,jkab->ij", mp.t2oo + td2, hf.oovv)
            - 1.0 * einsum("ikab,jkab->ij", mp.t2oo, t2eri_sum)
            + 2.0 * einsum("kija,ka->ij", hf.ooov, p0.ov)
        ).symmetrise()
        - einsum("ikjl,kl->ij", hf.oooo, p0.oo)
        - einsum("iajb,ab->ij", hf.ovov, p0.vv)
    )


def cvs_adc3_i2(hf, mp, intermediates):
    cvs_p0 = intermediates.cvs_p0
    return (
        + 2.0 * einsum("kIJa,ka->IJ", hf.occv, cvs_p0.ov).symmetrise()
        - 1.0 * einsum("kIlJ,kl->IJ", hf.ococ, cvs_p0.oo)
        - 1.0 * einsum("IaJb,ab->IJ", hf.cvcv, cvs_p0.vv)
    )


@register_as_intermediate
def adc3_m11(hf, mp, intermediates):
    td2 = mp.td2(b.oovv)
    p0 = mp.diffdm(2)

    i1 = adc3_i1(hf, mp, intermediates).evaluate()
    i2 = adc3_i2(hf, mp, intermediates).evaluate()
    t2sq = einsum("ikac,jkbc->iajb", mp.t2oo, mp.t2oo).evaluate()

    # Build two Kronecker deltas
    d_oo = zeros_like(hf.foo)
    d_vv = zeros_like(hf.fvv)
    d_oo.set_mask("ii", 1.0)
    d_vv.set_mask("aa", 1.0)

    t2eri_sum = (
        + 2.0 * mp.t2eri(b.oovv, b.ov).symmetrise((0, 1), (2, 3))  # t2eri4
        + 0.5 * mp.t2eri(b.oovv, b.vv)                             # t2eri5
        + 0.5 * mp.t2eri(b.oovv, b.oo)                             # t2eri3
    )
    return (
        + einsum("ij,ab->iajb", d_oo, hf.fvv + i1)
        - einsum("ij,ab->iajb", hf.foo - i2, d_vv)
        - einsum("jaib->iajb", hf.ovov)
        - (  # symmetrise i<>j and a<>b
            + einsum("jkbc,ikac->iajb", hf.oovv, mp.t2oo + td2)
            - einsum("jkbc,ikac->iajb", mp.t2oo, t2eri_sum)
            - einsum("ibac,jc->iajb", hf.ovvv, 2.0 * p0.ov)
            - einsum("ikja,kb->iajb", hf.ooov, 2.0 * p0.ov)
            - einsum("jaic,bc->iajb", hf.ovov, p0.vv)
            + einsum("ik,jakb->iajb", p0.oo, hf.ovov)
            + einsum("ibkc,kajc->iajb", hf.ovov, 2.0 * t2sq)
        ).symmetrise((0, 2), (1, 3))
        # TODO This hack is done to avoid opt_einsum being smart and instantiating
        #      a tensor of dimension 6 (to avoid the vvvv tensor) in some cases,
        #      which is the right thing to do, but not yet supported.
        # + 0.5 * einsum("icjd,klac,klbd->iajb", hf.ovov, mp.t2oo, mp.t2oo)
        + 0.5 * einsum("icjd,acbd->iajb", hf.ovov,
                       einsum("klac,klbd->acbd", mp.t2oo, mp.t2oo))
        # + 0.5 * einsum("ikcd,jlcd,kalb->iajb", mp.t2oo, mp.t2oo, hf.ovov)
        + 0.5 * einsum("ikjl,kalb->iajb",
                       einsum("ikcd,jlcd->ikjl", mp.t2oo, mp.t2oo), hf.ovov)
        - einsum("iljk,kalb->iajb", hf.oooo, t2sq)
        - einsum("idjc,acbd->iajb", t2sq, hf.vvvv)
    )


@register_as_intermediate
def cvs_adc3_m11(hf, mp, intermediates):
    i1 = adc3_i1(hf, mp, intermediates).evaluate()
    i2 = cvs_adc3_i2(hf, mp, intermediates).evaluate()
    t2sq = einsum("ikac,jkbc->iajb", mp.t2oo, mp.t2oo).evaluate()

    # Build two Kronecker deltas
    d_cc = zeros_like(hf.fcc)
    d_vv = zeros_like(hf.fvv)
    d_cc.set_mask("II", 1.0)
    d_vv.set_mask("aa", 1.0)

    return (
        + einsum("IJ,ab->IaJb", d_cc, hf.fvv + i1)
        - einsum("IJ,ab->IaJb", hf.fcc - i2, d_vv)
        - einsum("JaIb->IaJb", hf.cvcv)
        + (  # symmetrise I<>J and a<>b
            + einsum("JaIc,bc->IaJb", hf.cvcv, intermediates.cvs_p0.vv)
            - einsum("kIJa,kb->IaJb", hf.occv, 2.0 * intermediates.cvs_p0.ov)
        ).symmetrise((0, 2), (1, 3))
        # TODO This hack is done to avoid opt_einsum being smart and instantiating
        #      a tensor of dimension 6 (to avoid the vvvv tensor) in some cases,
        #      which is the right thing to do, but not yet supported.
        # + 0.5 * einsum("IcJd,klac,klbd->IaJb", hf.cvcv, mp.t2oo, mp.t2oo)
        + 0.5 * einsum("IcJd,acbd->IaJb", hf.cvcv,
                       einsum("klac,klbd->acbd", mp.t2oo, mp.t2oo))
        - einsum("lIkJ,kalb->IaJb", hf.ococ, t2sq)
    )


@register_as_intermediate
def adc3_pia(hf, mp, intermediates):
    # This definition differs from libadc. It additionally has the hf.ooov term.
    return (                          # Perturbation theory in ADC coupling block
        + hf.ooov                                            # 1st order
        - 2.0 * mp.t2eri(b.ooov, b.ov).antisymmetrise(0, 1)  # 2nd order
        - 0.5 * mp.t2eri(b.ooov, b.vv)                       # 2nd order
    )


@register_as_intermediate
def cvs_adc3_pia(hf, mp, intermediates):
    # Perturbation theory in CVS-ADC coupling block:
    #       1st                     2nd
    return hf.occv - einsum("jlac,lKIc->jIKa", mp.t2oo, hf.occv)


@register_as_intermediate
def adc3_pib(hf, mp, intermediates):
    # This definition differs from libadc. It additionally has the hf.ovvv term.
    return (                          # Perturbation theory in ADC coupling block
        + hf.ovvv                                            # 1st order
        + 2.0 * mp.t2eri(b.ovvv, b.ov).antisymmetrise(2, 3)  # 2nd order
        - 0.5 * mp.t2eri(b.ovvv, b.oo)                       # 2nd order
    )
