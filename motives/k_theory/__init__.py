from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow

from motives.core.lambda_ring_expr import LambdaRingExpr

from .chern import chern_character, chern_class
from .objects import Curve, Scheme, VectorBundle, Wedge, SymPower, DeterminantBundle, DualBundle
from .operations import Det, Dual, End, Hom, sym, to_sym, to_wedge, wedge, exact_sequence_realtions, solve_exact_sequence

LambdaRingExpr.to_wedge = to_wedge
LambdaRingExpr.to_sym = to_sym
LambdaRingExpr.wedge = wedge
LambdaRingExpr.sym = sym
LambdaRingExpr.c = chern_class
LambdaRingExpr.ch = chern_character

Add.to_wedge = to_wedge
Add.to_sym = to_sym
Add.wedge = wedge
Add.sym = sym
Add.c = chern_class
Add.ch = chern_character

Mul.to_wedge = to_wedge
Mul.to_sym = to_sym
Mul.wedge = wedge
Mul.sym = sym
Mul.c = chern_class
Mul.ch = chern_character

Pow.to_wedge = to_wedge
Pow.to_sym = to_sym
Pow.wedge = wedge
Pow.sym = sym
Pow.c = chern_class
Pow.ch = chern_character