import pytest
import motives as mtv


def test_expr_free():
    x = mtv.Free("x")
    y = mtv.Free("y")
    z = mtv.Free("z")
    t = mtv.Free("t")

    et = (
        (x + y**3).adams(5) / (2 * z).adams(4).lambda_(12) - (t**2 * z - 1 - x).sigma(8)
    ).adams(3) * ((y + z) ** 2 - 3).sigma(4)

    assert et.to_adams(as_symbol=True) == et.to_adams(as_symbol=False).to_adams(
        as_symbol=True
    )


def test_expr_motives():
    cur1 = mtv.Curve("x", 5)
    cur2 = mtv.SO(6)
    bund = mtv.BunDet(cur1, 4)
    t = mtv.Free("t")
    proj = mtv.Proj(12)

    et = (
        (cur1 + cur2**3).adams(5) / (2 * bund).lambda_(6)
        - (t**2 * bund - 1 - cur1).sigma(6)
    ).adams(3) * ((cur2 + bund) ** 2 - 3).adams(3).sigma(5) + proj.adams(2)

    adams = et.to_adams(as_symbol=True)
    adams_double1 = et.to_adams(as_symbol=False)
    adams_double = adams_double1.to_adams(as_symbol=True)

    assert adams == adams_double
