import sympy as sp
import math
from functools import reduce

from .operand import Operand


def get_max_adams_degree_num(self: sp.Rational) -> int:
    """
    Computes the maximum Adams degree of this tree for `Rational` nodes.

    This function computes the maximum Adams degree for `Rational` expressions.
    Since `Rational` nodes do not contribute to Adams degrees, the result is always 1.

    This function is intended to be added as a method for `sp.Rational`.

    Returns:
    --------
    int
        The maximum Adams degree of this tree once converted to an Adams polynomial, always 1.
    """
    return 1


def get_max_groth_degree_num(self: sp.Rational) -> int:
    """
    Computes the degree required to create a Grothendieck context for `Rational` nodes.

    This function calculates the maximum sigma or lambda degree for `Rational` expressions.
    Since `Rational` nodes do not contribute to Grothendieck degrees, the result is always 0.

    This function is intended to be added as a method for `sp.Rational`.

    Returns:
    --------
    int
        The maximum sigma or lambda degree of this tree, always 0.
    """
    return 0


def get_adams_var_num(self: sp.Rational, i: int) -> sp.Expr:
    """
    Returns the `Rational` operand with an Adams operation applied to it.

    This function simply returns the `Rational` operand itself, as Adams operations
    do not modify `Rational` nodes.

    This function is intended to be added as a method for `sp.Rational`.

    Args:
    -----
    i : int
        The degree of the Adams operator.

    Returns:
    --------
    sp.Expr
        The `Rational` operand itself with no changes.
    """
    return self


def get_lambda_var_num(self: sp.Rational, i: int) -> sp.Expr:
    """
    Returns the `Rational` operand with a lambda operation applied to it.

    This function computes the result of applying the lambda operator to a `Rational`
    operand. It calculates a new `Rational` expression based on the degree of the lambda operator.

    This function is intended to be added as a method for `sp.Rational`.

    Args:
    -----
    i : int
        The degree of the lambda operator.

    Returns:
    --------
    sp.Expr
        The `Rational` operand with the lambda operator applied, expressed as a `Rational` value.
    """
    return sp.Rational(
        reduce(lambda x, y: x * y, (self.p + j * self.q for j in range(i))),
        math.factorial(i) * self.q**i,
    )


def _to_adams_num(
    self: sp.Rational,
    operands: set[Operand],
    max_adams_degree: int,
    as_symbol: bool = False,
) -> sp.Expr:
    """
    Converts this `Rational` subtree into an equivalent Adams polynomial.

    Since `Rational` expressions are unaffected by Adams operations, this function
    returns the operand itself.

    This function is intended to be added as a method for `sp.Rational`.

    Args:
    -----
    operands : set[Operand]
        The set of all operands in the expression tree.
    max_adams_degree : int
        The maximum Adams degree in the expression.
    as_symbol : bool, optional
        Whether to represent the Adams operators as symbols. Defaults to False.

    Returns:
    --------
    sp.Expr
        The `Rational` operand itself.
    """
    return self


def _to_adams_lambda_num(
    self: sp.Rational,
    operands: set[Operand],
    max_adams_degree: int,
    as_symbol: bool = False,
    adams_degree: int = 1,
) -> sp.Expr:
    """
    Converts this `Rational` subtree into an equivalent Adams polynomial, optimized for lambda conversion.

    This function is similar to `_to_adams_num` but optimized for lambda conversions. Since
    `Rational` expressions are unaffected by Adams operations, the function simply returns the operand.

    This function is intended to be added as a method for `sp.Rational`.

    Args:
    -----
    operands : set[Operand]
        The set of all operands in the expression tree.
    max_adams_degree : int
        The maximum Adams degree in the expression.
    as_symbol : bool, optional
        Whether to represent the Adams operators as symbols. Defaults to False.
    adams_degree : int, optional
        The cumulative Adams degree higher than this node in the expression tree. Defaults to 1.

    Returns:
    --------
    sp.Expr
        The `Rational` operand itself.
    """
    return self._to_adams(operands, max_adams_degree, as_symbol)


def _subs_adams_num(
    self: sp.Rational, ph: sp.Expr, max_adams_degree: int, as_symbol: bool = False
) -> sp.Expr:
    """
    Substitutes any Adams of the `Rational` operand in `ph` for its equivalent polynomial of lambdas.

    Since `Rational` nodes do not generate Adams variables, this function simply returns
    the original polynomial without changes. This method is called in `to_lambda` to substitute
    Adams expressions in the polynomial.

    This function is intended to be added as a method for `sp.Rational`.

    Args:
    -----
    ph : sp.Expr
        The polynomial to substitute the Adams variables into.
    max_adams_degree : int
        The maximum Adams degree in the expression.
    as_symbol : bool, optional
        Whether to represent the Adams operators as symbols. Defaults to False.

    Returns:
    --------
    sp.Expr
        The polynomial `ph`, unchanged.
    """
    return ph
