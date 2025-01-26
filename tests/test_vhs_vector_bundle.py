import pytest
from motives import VHS, Curve, Lefschetz, VectorBundleModuli, TwistedHiggsModuliBB
import math


@pytest.mark.parametrize(
    "g,p,rks",
    (
        (g, p, rks)
        for g in range(2, 6)
        for p in range(2, 6)
        for rks in [(1, 1), (1, 2), (2, 1), (1, 1, 1)]
    ),
)
def test_vhs(g, p, rks):
    dl = 2 * g - 2 + p
    end_range = (
        math.floor(dl / 2 + 1 / 3) + 1
        if rks == (1, 2)
        else ((dl + 1) // 2 + 1 if rks == (1, 1) else math.floor(dl / 2 + 2 / 3) + 1)
    )

    cur = Curve("c", g)
    lef = Lefschetz()

    thm = TwistedHiggsModuliBB(cur, p, 2)
    thm._initiate_vhs()

    vhs_thm = thm.vhs[rks]

    vhs_real = 0
    for i in range(1, end_range):
        vhs = VHS(cur, rks, (i, -i + 1), p)._et_repr
        vhs_real += vhs

    vhs_real = vhs_real * lef ** (3 * dl + 2 - 2 * g)

    assert (vhs_thm - vhs_real).to_adams(as_symbol=False).simplify() == 0


@pytest.mark.parametrize("g,r", ((g, r) for g in range(2, 6) for r in [2, 3]))
def test_vector_bundle(g, r):
    d = 1

    cur = Curve("c", g)
    vbm = VectorBundleModuli(cur, r, d)

    if r == 2:
        et1 = vbm._compute_motive_rk2().to_adams(as_symbol=False)
    elif r == 3:
        et1 = vbm._compute_motive_rk3().to_adams(as_symbol=False)
    et2 = vbm._compute_motive_rkr(r, d).to_adams(as_symbol=False)

    result = (et1 - et2).simplify()

    assert result == 0


@pytest.mark.parametrize(
    "g,d,r",
    (
        (g, d, r)
        for g in range(2, 5)
        for (d, r) in [(1, 2), (1, 3), (1, 4), (1, 5), (2, 5)]
    ),
)
def test_vector_bundle_r(g, d, r):
    cur = Curve("c", g)

    vbm = VectorBundleModuli(cur, r, d)
    et1 = vbm._compute_motive_rkr(r, d).to_adams(as_symbol=False)

    vbm = VectorBundleModuli(cur, r, d + r)
    et2 = vbm._compute_motive_rkr(r, d + r).to_adams(as_symbol=False)

    vbm = VectorBundleModuli(cur, r, d + 2 * r)
    et3 = vbm._compute_motive_rkr(r, d + 2 * r).to_adams(as_symbol=False)

    vbm = VectorBundleModuli(cur, r, r - d)
    et4 = vbm._compute_motive_rkr(r, r - d).to_adams(as_symbol=False)

    assert (et1 - et2).simplify() == 0
    assert (et2 - et3).simplify() == 0
    assert (et3 - et4).simplify() == 0
