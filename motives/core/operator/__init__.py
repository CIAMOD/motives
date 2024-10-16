from .nary_operator import _to_adams_lambda_nary, _to_adams_nary, get_max_adams_degree_nary, get_max_groth_degree_nary
from .general_operator import to_adams, to_lambda, sigma, lambda_, adams
from .pow_operator import _to_adams_lambda_pow, _to_adams_pow, get_max_adams_degree_pow, get_max_groth_degree_pow
from .rational_operator import get_max_adams_degree_num, get_max_groth_degree_num, get_adams_var_num, get_lambda_var_num, _to_adams_num, _to_adams_lambda_num, _subs_adams_num

import sympy

sympy.core.add.Add.to_adams = to_adams
sympy.core.add.Add.to_lambda = to_lambda
sympy.core.add.Add.sigma = sigma
sympy.core.add.Add.lambda_ = lambda_
sympy.core.add.Add.adams = adams
sympy.core.add.Add.get_max_adams_degree = get_max_adams_degree_nary
sympy.core.add.Add.get_max_groth_degree = get_max_groth_degree_nary
sympy.core.add.Add._to_adams = _to_adams_nary
sympy.core.add.Add._to_adams_lambda = _to_adams_lambda_nary

sympy.core.mul.Mul.to_adams = to_adams
sympy.core.mul.Mul.to_lambda = to_lambda
sympy.core.mul.Mul.sigma = sigma
sympy.core.mul.Mul.lambda_ = lambda_
sympy.core.mul.Mul.adams = adams
sympy.core.mul.Mul.get_max_adams_degree = get_max_adams_degree_nary
sympy.core.mul.Mul.get_max_groth_degree = get_max_groth_degree_nary
sympy.core.mul.Mul._to_adams = _to_adams_nary
sympy.core.mul.Mul._to_adams_lambda = _to_adams_lambda_nary

sympy.core.power.Pow.to_adams = to_adams
sympy.core.power.Pow.to_lambda = to_lambda
sympy.core.power.Pow.sigma = sigma
sympy.core.power.Pow.lambda_ = lambda_
sympy.core.power.Pow.adams = adams
sympy.core.power.Pow.get_max_adams_degree = get_max_adams_degree_pow
sympy.core.power.Pow.get_max_groth_degree = get_max_groth_degree_pow
sympy.core.power.Pow._to_adams = _to_adams_pow
sympy.core.power.Pow._to_adams_lambda = _to_adams_lambda_pow

sympy.core.numbers.Rational.to_adams = to_adams
sympy.core.numbers.Rational.to_lambda = to_lambda
sympy.core.numbers.Rational.sigma = sigma
sympy.core.numbers.Rational.lambda_ = lambda_
sympy.core.numbers.Rational.adams = adams
sympy.core.numbers.Rational.get_max_adams_degree = get_max_adams_degree_num
sympy.core.numbers.Rational.get_max_groth_degree = get_max_groth_degree_num
sympy.core.numbers.Rational.get_adams_var = get_adams_var_num
sympy.core.numbers.Rational.get_lambda_var = get_lambda_var_num
sympy.core.numbers.Rational._to_adams = _to_adams_num
sympy.core.numbers.Rational._to_adams_lambda = _to_adams_lambda_num
sympy.core.numbers.Rational._subs_adams = _subs_adams_num