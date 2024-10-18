import pytest
import sympy as sp

from motives import Curve, Lefschetz, Point, CurveHodge, Lambda_
from motives.utils import partitions

@pytest.mark.parametrize("k", [k for k in range(1, 9)])
def test_lambda_of_adams_expansion(k: int) -> None:
    g = 3

    et = Curve("C", g=g).lambda_(k)

    point = Point()
    le = Lefschetz()
    hodge = CurveHodge("C", g=g)

    et_conv = sp.Add(
        *[
            sp.Mul(*[Lambda_(i1, point), Lambda_(i2, le), Lambda_(i3, hodge)])
            for (i1, i2, i3) in partitions(k, 3)
        ]
    )

    assert (et.to_adams() - et_conv.to_adams()).simplify() == 0
    return


@pytest.mark.parametrize("g,k", [(g, k) for k in range(1, 10) for g in range(2, 5)])
def test_hodge_lambda(g: int, k: int) -> None:
    g = 3
    h = CurveHodge("C", g=g)

    et = h.lambda_(k)

    pol = et.to_lambda()

    diff = pol - h.get_lambda_var(k)

    assert diff == 0 or diff.expand().simplify() == 0


def test_curve_lambda() -> None:
    g = 2
    cur = Curve("C", g=g)

    et = cur.lambda_(3)

    pol_wo_lambda = et.to_lambda(optimize=False).expand().simplify()
    pol_w_lambda = et.to_lambda(optimize=True).expand().simplify()

    assert (pol_wo_lambda - pol_w_lambda).expand().simplify() == 0
