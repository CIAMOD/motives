import pytest
import sympy as sp

from motives import Curve, Lefschetz, Point, CurveChow, Lambda_
from motives.utils import partitions


@pytest.mark.parametrize("k", [k for k in range(1, 9)])
def test_lambda_of_adams_expansion(k: int) -> None:
    g = 3

    et = Curve("C", g=g).lambda_(k)

    point = Point()
    le = Lefschetz()
    chow = CurveChow("C", g=g)

    et_conv = sp.Add(
        *[
            sp.Mul(*[Lambda_(i1, point), Lambda_(i2, le), Lambda_(i3, chow)])
            for (i1, i2, i3) in partitions(k, 3)
        ]
    )

    assert (
        et.to_adams(as_symbol=False) - et_conv.to_adams(as_symbol=False)
    ).simplify() == 0
    return


@pytest.mark.parametrize("g,k", [(g, k) for k in range(1, 10) for g in range(2, 5)])
def test_chow_lambda(g: int, k: int) -> None:
    g = 3
    h = CurveChow("C", g=g)

    et = h.lambda_(k)

    pol = et.to_lambda(as_symbol=False)

    diff = pol - h.get_lambda_var(k, as_symbol=False)

    assert diff == 0 or diff.expand().simplify() == 0


def test_curve_lambda() -> None:
    g = 2
    cur = Curve("C", g=g)

    et = cur.lambda_(3)

    pol_wo_lambda = et.to_lambda(as_symbol=False, optimize=False).expand().simplify()
    pol_w_lambda = et.to_lambda(as_symbol=False, optimize=True).expand().simplify()

    assert (pol_wo_lambda - pol_w_lambda).expand().simplify() == 0
