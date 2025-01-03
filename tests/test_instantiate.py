# -------- BASE ----------
from motives import Free

# -------- CORE ----------
from motives import LambdaRingContext
from motives import LambdaRingExpr
from motives import Operand
from motives import Object1Dim

# -------- OPERATOR ----------
from motives import Lambda_, Sigma, Adams

# -------- GROTHENDIECK ----------
from motives import Motive
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
from motives import TwistedHiggsModuli
from motives import (
    TwistedHiggsModuliBB,
)
from motives import (
    TwistedHiggsModuliADHM,
)
from motives import VHS
from motives import VectorBundleModuli

### -------- STACK ----------
from motives import BunG
from motives import BunDet

import pytest


def test_instantiate() -> None:
    free = Free("x")
    lambda_ring_context = LambdaRingContext()
    point = Point()
    polynomial_1_var = Polynomial1Var("a")
    proj = Proj(5)
    lefschetz = Lefschetz()
    curve = Curve("a", 5)
    curve_chow = CurveChow("a", 5)
    jacobian = Jacobian(curve)
    gl = GL(4)
    g = SemisimpleG([1, 2, 3], 4)
    psl = PSL(4)
    pgl = PGL(4)
    sl = SL(4)
    so_even = SO(4)
    so_odd = SO(5)
    sp = SP(4)
    a = A(12)
    b = B(12)
    c = C(12)
    d = D(12)
    e = E(6)
    f4 = F4()
    BSL= SL(3).BG()
    twisted_higgs_moduli = TwistedHiggsModuli(curve, 5, 3)
    twisted_higgs_moduli_bb = TwistedHiggsModuliBB(curve, 5, 3)
    twisted_higgs_moduli_adhm = TwistedHiggsModuliADHM(curve, 5, 3)
    vhs = VHS(curve, (1,2),(1,1),1)
    vector_bundle_moduli = VectorBundleModuli(curve, 3,2)
    bun_g = BunG(curve, sp)
    bun = BunDet(curve, 5)
