import pytest

from motives import Curve, Lefschetz
import sympy as sp


@pytest.mark.parametrize("g", [g for g in range(2, 6)])
def test_sym(g: int) -> None:
    cur = Curve("X", g=g)
    sym = cur.Sym(2)
    L = Lefschetz()
    real = (
        1 + cur.curve_chow.lambda_(2) + L**2 + cur.curve_chow + L + L * cur.curve_chow
    )
    assert (sym - real).to_lambda(as_symbol=False).simplify() == 0
    return


@pytest.mark.parametrize("g", [g for g in range(2, 6)])
def test_alt(g: int) -> None:
    cur = Curve("X", g=g)
    alt = cur.Alt(2)
    L = Lefschetz()
    assert L.sigma(2).to_lambda(as_symbol=False) == 0
    assert sp.Integer(1).sigma(2).to_lambda(as_symbol=False) == 0
    real = cur.curve_chow.sigma(2) + cur.curve_chow + L + L * cur.curve_chow
    assert (alt - real).to_lambda(as_symbol=False).simplify() == 0
    return
