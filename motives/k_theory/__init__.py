from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow

from motives.core.lambda_ring_expr import LambdaRingExpr

from .scheme import Scheme, Curve
from .vector_bundle import VectorBundle
from .operations import to_wedge, to_sym
from .power_operations import Wedge, SymPower


LambdaRingExpr.to_wedge = to_wedge
LambdaRingExpr.to_sym = to_sym

Add.to_wedge = to_wedge
Add.to_sym = to_sym

Mul.to_wedge = to_wedge
Mul.to_sym = to_sym

Pow.to_wedge = to_wedge
Pow.to_sym = to_sym