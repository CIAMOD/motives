import pytest
import sympy as sp

from motives import TwistedHiggsModuliADHM, Curve, Lefschetz
from motives.equations import compare_equation


def test_mozgovoy222() -> None:
    x = Curve("x", 2)

    moz = TwistedHiggsModuliADHM(x, p=2, r=2)
    eq = moz.moz_lambdas(verbose=0)

    lef = Lefschetz()
    a = x.curve_chow.lambda_symbols

    domain = sp.ZZ[[lef] + a[1:]]

    solved_eq = domain.from_sympy(
        lef**10
        * (lef**2 + a[1] * lef + a[1] + a[2] + 1)
        * (
            2 * lef
            + 2 * a[1]
            + a[2]
            + 2 * lef * a[1]
            + lef * a[2]
            + lef**2 * a[1]
            + lef**3 * a[1]
            + 2 * lef**2
            + 2 * lef**3
            + lef**4
            + lef**5
            + 2
        )
    )

    assert (eq - solved_eq) == 0
    return


def test_mozgovoy232() -> None:
    x = Curve("x", 2)

    moz = TwistedHiggsModuliADHM(x, p=3, r=2)
    eq = moz.moz_lambdas(verbose=0)

    lef = Lefschetz()
    a = x.curve_chow.lambda_symbols

    domain = sp.ZZ[[lef] + a[1:]]

    solved_eq = domain.from_sympy(
        lef**13
        * (lef**2 + a[1] * lef + a[1] + a[2] + 1)
        * (
            2 * lef
            + 2 * a[1]
            + 2 * a[2]
            + 3 * lef * a[1]
            + lef * a[2]
            + 2 * lef**2 * a[1]
            + lef**2 * a[2]
            + lef**3 * a[1]
            + lef**4 * a[1]
            + 3 * lef**2
            + 2 * lef**3
            + 2 * lef**4
            + lef**5
            + lef**6
            + 3
        )
    )

    assert (eq - solved_eq) == 0
    return


def test_mozgovoy242() -> None:
    x = Curve("x", 2)

    moz = TwistedHiggsModuliADHM(x, p=4, r=2)
    eq = moz.moz_lambdas(verbose=0)

    lef = Lefschetz()
    a = x.curve_chow.lambda_symbols

    domain = sp.ZZ[[lef] + a[1:]]

    solved_eq = domain.from_sympy(
        lef**16
        * (lef**2 + a[1] * lef + a[1] + a[2] + 1)
        * (
            3 * lef
            + 3 * a[1]
            + 2 * a[2]
            + 4 * lef * a[1]
            + 2 * lef * a[2]
            + 3 * lef**2 * a[1]
            + lef**2 * a[2]
            + 2 * lef**3 * a[1]
            + lef**3 * a[2]
            + lef**4 * a[1]
            + lef**5 * a[1]
            + 3 * lef**2
            + 3 * lef**3
            + 2 * lef**4
            + 2 * lef**5
            + lef**6
            + lef**7
            + 3
        )
    )

    assert (eq - solved_eq) == 0
    return


def test_mozgovoy213() -> None:
    x = Curve("x", 2)

    moz = TwistedHiggsModuliADHM(x, p=1, r=3)
    eq = moz.moz_lambdas(verbose=0)

    lef = Lefschetz()
    a = x.curve_chow.lambda_symbols

    domain = sp.ZZ[[lef] + a[1:]]

    solved_eq = domain.from_sympy(
        lef**15
        * (lef**2 + a[1] * lef + a[1] + a[2] + 1)
        * (
            lef**11
            + lef**10
            + lef**9 * a[1]
            + 3 * lef**9
            + 2 * lef**8 * a[1]
            + 4 * lef**8
            + 4 * lef**7 * a[1]
            + lef**7 * a[2]
            + 7 * lef**7
            + lef**6 * a[1] ** 2
            + 8 * lef**6 * a[1]
            + lef**6 * a[2]
            + 9 * lef**6
            + lef**5 * a[1] ** 2
            + 12 * lef**5 * a[1]
            + 4 * lef**5 * a[2]
            + 14 * lef**5
            + 3 * lef**4 * a[1] ** 2
            + lef**4 * a[1] * a[2]
            + 18 * lef**4 * a[1]
            + 5 * lef**4 * a[2]
            + 15 * lef**4
            + 5 * lef**3 * a[1] ** 2
            + 2 * lef**3 * a[1] * a[2]
            + 22 * lef**3 * a[1]
            + 9 * lef**3 * a[2]
            + 18 * lef**3
            + 6 * lef**2 * a[1] ** 2
            + 4 * lef**2 * a[1] * a[2]
            + 24 * lef**2 * a[1]
            + 10 * lef**2 * a[2]
            + 15 * lef**2
            + 6 * lef * a[1] ** 2
            + 5 * lef * a[1] * a[2]
            + 18 * lef * a[1]
            + lef * a[2] ** 2
            + 10 * lef * a[2]
            + 12 * lef
            + 3 * a[1] ** 2
            + 4 * a[1] * a[2]
            + 9 * a[1]
            + a[2] ** 2
            + 6 * a[2]
            + 6
        )
    )

    assert (eq - solved_eq) == 0
    return


@pytest.mark.parametrize(
    # "g,p,r",  [(i, j, k) for i in range(2, 8) for j in range(2, 9) for k in range(2, 4)]
    "g,p,r",
    [(2, 2, 2), (5, 8, 3)],
)
def test_mozgovoy_equal(g: int, p: int, r: int) -> None:
    assert compare_equation(g, p, r, f"data/{g}_{p}_{r}.p", verbose=0) is True
    return
