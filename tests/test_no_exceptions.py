import pytest
import sympy as sp

from motives import Free

# -------- GROTHENDIECK ----------
from motives import Point
from motives import Polynomial1Var
from motives import Proj
from motives import Lefschetz

## -------- CURVES ----------
from motives import Curve
from motives import CurveChow
from motives import Jacobian

## -------- GL ----------
from motives import GL
from motives import SemisimpleG
from motives import PSL
from motives import PGL
from motives import SL
from motives import SO
from motives import SP
from motives import A, B, C, D, E, F4

## -------- MODULI ----------
### -------- SCHEME ----------
from motives import VHS
from motives import VectorBundleModuli

### -------- STACK ----------
from motives import BunG
from motives import BunDet


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

    et = et.to_lambda(as_symbol=False)
    return


def test_adams() -> None:
    x = Free("x")
    y = Free("y")
    z = Free("z")

    et = ((z - x * y**4).adams(3) + (x + y).lambda_(4) - z.adams(5)).sigma(2) - 1 / (
        (x + y) ** 3
    ).lambda_(3)

    et = et.to_adams(as_symbol=False)
    return


def test_adams_lambda_all() -> None:
    free = Free("x")
    point = Point()
    polynomial = Polynomial1Var("t")
    proj = Proj(4)
    lefschetz = Lefschetz()
    curve = Curve("C", 4)
    curve_chow = CurveChow("C", 4)
    jacobian = Jacobian(curve)
    gl = GL(4)
    g = SemisimpleG([4, 3, 4, 8], 19)
    psl = PSL(4)
    pgl = PGL(4)
    sl = SL(4)
    so_odd = SO(3)
    so_even = SO(4)
    sp = SP(4)
    a = A(4)
    b = B(4)
    c = C(3)
    d = D(4)
    e = E(6)
    f4 = F4()
    vhs = VHS(curve, (2, 1), (1, 1), 3).calculate()
    vector_bundle_moduli = VectorBundleModuli(curve, 3, 2).calculate()
    bun_g = BunG(curve, psl)
    bun = BunDet(curve, 4)

    ets = [
        free,
        point,
        polynomial,
        proj,
        lefschetz,
        curve,
        curve_chow,
        jacobian,
        gl,
        g,
        psl,
        pgl,
        sl,
        so_odd,
        so_even,
        sp,
        a,
        b,
        c,
        d,
        e,
        f4,
        vhs,
        vector_bundle_moduli,
        bun_g,
        bun,
    ]
    for el in ets:
        el.lambda_(3).to_adams(as_symbol=False)
        el.sigma(3).to_lambda(as_symbol=False)
