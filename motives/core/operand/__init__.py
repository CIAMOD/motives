from .rational_operand import (
    get_max_adams_degree_num,
    get_max_groth_degree_num,
    get_adams_var_num,
    get_lambda_var_num,
    _to_adams_num,
    _to_adams_lambda_num,
    _subs_adams_num,
)
import sympy as sp
from ..operator.ring_operator import to_adams, to_lambda, sigma, lambda_, adams


sp.core.numbers.Rational.get_max_adams_degree = get_max_adams_degree_num
sp.core.numbers.Rational.get_max_groth_degree = get_max_groth_degree_num
sp.core.numbers.Rational.get_adams_var = get_adams_var_num
sp.core.numbers.Rational.get_lambda_var = get_lambda_var_num
sp.core.numbers.Rational._to_adams = _to_adams_num
sp.core.numbers.Rational._to_adams_lambda = _to_adams_lambda_num
sp.core.numbers.Rational._subs_adams = _subs_adams_num

sp.core.numbers.Rational.to_adams = to_adams
sp.core.numbers.Rational.to_lambda = to_lambda
sp.core.numbers.Rational.sigma = sigma
sp.core.numbers.Rational.lambda_ = lambda_
sp.core.numbers.Rational.adams = adams
