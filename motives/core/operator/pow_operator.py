import sympy as sp

from ..operand.operand import Operand


def get_max_adams_degree_pow(self: sp.Pow) -> int:
    """
    Computes the maximum Adams degree of this expression tree for `Pow` nodes.

    This function calculates the maximum Adams degree of the `Pow` expression by
    traversing its base. It multiplies the Adams, lambda, and sigma degrees of the base
    and returns the maximum value.

    This function is intended to be added as a method for `sp.Pow`.

    Returns:
    --------
    int
        The maximum Adams degree of the tree after conversion to an Adams polynomial.
    """
    return self.base.get_max_adams_degree()


def get_max_groth_degree_pow(self: sp.Pow) -> int:
    """
    Computes the maximum degree required to create a Grothendieck context for `Pow` nodes.

    This function calculates the maximum sigma or lambda degree of the `Pow` expression
    by traversing its base and finding the required degree.

    This function is intended to be added as a method for `sp.Pow`.

    Returns:
    --------
    int
        The maximum sigma or lambda degree required for the tree.
    """
    return self.base.get_max_groth_degree()


def _to_adams_pow(
    self: sp.Pow, operands: set[Operand], max_adams_degree: int, as_symbol: bool = False
) -> sp.Expr:
    """
    Converts the `Pow` subtree into an equivalent Adams polynomial.

    This function converts the base of the `Pow` expression into its equivalent Adams
    polynomial, then raises the result to the exponent of the `Pow` node.

    This function is intended to be added as a method for `sp.Pow`.

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
        A polynomial of Adams operators equivalent to this subtree.
    """
    return self.base._to_adams(operands, max_adams_degree, as_symbol) ** self.exp


def _to_adams_lambda_pow(
    self: sp.Pow,
    operands: set[Operand],
    max_adams_degree: int,
    as_symbol: bool = False,
    adams_degree: int = 1,
) -> sp.Expr:
    """
    Converts the `Pow` subtree into an equivalent Adams polynomial, optimized for lambda conversion.

    This function is similar to `_to_adams_pow`, but it is optimized for lambda conversions
    when called from `to_lambda`. The base of the `Pow` expression is converted into an Adams
    polynomial using the lambda-optimized pathway, and the result is raised to the exponent.

    This function is intended to be added as a method for `sp.Pow`.

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
        A polynomial of Adams operators equivalent to this subtree.
    """
    return (
        self.base._to_adams_lambda(operands, max_adams_degree, as_symbol, adams_degree)
        ** self.exp
    )
