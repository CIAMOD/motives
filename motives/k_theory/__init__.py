from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow

from motives.core.lambda_ring_expr import LambdaRingExpr

from .scheme import Scheme, Curve
from .vector_bundle import VectorBundle
from .operations import to_wedge, to_sym, wedge, sym
from .power_operations import Wedge, SymPower
from .chern_character import chern_character

LambdaRingExpr.to_wedge = to_wedge
LambdaRingExpr.to_sym = to_sym
LambdaRingExpr.wedge = wedge
LambdaRingExpr.sym = sym

Add.to_wedge = to_wedge
Add.to_sym = to_sym
Add.wedge = wedge
Add.sym = sym

Mul.to_wedge = to_wedge
Mul.to_sym = to_sym
Mul.wedge = wedge
Mul.sym = sym

Pow.to_wedge = to_wedge
Pow.to_sym = to_sym
Pow.wedge = wedge
Pow.sym = sym