from .nary_operator import (
    _to_adams_lambda_nary,
    _to_adams_nary,
    get_max_adams_degree_nary,
    get_max_groth_degree_nary,
)
from .pow_operator import (
    _to_adams_lambda_pow,
    _to_adams_pow,
    get_max_adams_degree_pow,
    get_max_groth_degree_pow,
)
from .ring_operator import (
    sigma,
    lambda_,
    adams,
    to_adams,
    to_lambda,
    Lambda_,
    Sigma,
    Adams,
)

import sympy

sympy.core.add.Add.get_max_adams_degree = get_max_adams_degree_nary
sympy.core.add.Add.get_max_groth_degree = get_max_groth_degree_nary
sympy.core.add.Add._to_adams = _to_adams_nary
sympy.core.add.Add._to_adams_lambda = _to_adams_lambda_nary

sympy.core.mul.Mul.get_max_adams_degree = get_max_adams_degree_nary
sympy.core.mul.Mul.get_max_groth_degree = get_max_groth_degree_nary
sympy.core.mul.Mul._to_adams = _to_adams_nary
sympy.core.mul.Mul._to_adams_lambda = _to_adams_lambda_nary

sympy.core.power.Pow.get_max_adams_degree = get_max_adams_degree_pow
sympy.core.power.Pow.get_max_groth_degree = get_max_groth_degree_pow
sympy.core.power.Pow._to_adams = _to_adams_pow
sympy.core.power.Pow._to_adams_lambda = _to_adams_lambda_pow


sympy.core.add.Add.to_adams = to_adams
sympy.core.add.Add.to_lambda = to_lambda
sympy.core.add.Add.sigma = sigma
sympy.core.add.Add.lambda_ = lambda_
sympy.core.add.Add.adams = adams

sympy.core.mul.Mul.to_adams = to_adams
sympy.core.mul.Mul.to_lambda = to_lambda
sympy.core.mul.Mul.sigma = sigma
sympy.core.mul.Mul.lambda_ = lambda_
sympy.core.mul.Mul.adams = adams

sympy.core.power.Pow.to_adams = to_adams
sympy.core.power.Pow.to_lambda = to_lambda
sympy.core.power.Pow.sigma = sigma
sympy.core.power.Pow.lambda_ = lambda_
sympy.core.power.Pow.adams = adams
