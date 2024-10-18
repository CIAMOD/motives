import pytest
import sympy as sp

from motives import Free

def test_construction() -> None:
    x = Free("x")
    y = Free("y")
    z = Free("z")

    et = (
        (x + (y + z).adams(4)) ** 3
        - (((x * y + z).sigma(4) / (y - z.adams(3) + 3)) ** 2).lambda_(7)
        + x * y * z
        - 1
    )
    return


def test_lambda() -> None:
    x = Free("x")
    y = Free("y")
    z = Free("z")

    et = (
        (x / y).adams(3)
        - ((y * z).sigma(4) ** 2 + (z / x) + sp.Integer(3) / 5).lambda_(5)
        + 1
    )

    et = et.to_lambda()
    return


def test_adams() -> None:
    x = Free("x")
    y = Free("y")
    z = Free("z")

    et = ((z - x * y**4).adams(3) + (x + y).lambda_(4) - z.adams(5)).sigma(2) - 1 / (
        (x + y) ** 3
    ).lambda_(3)

    et = et.to_adams()
    return
