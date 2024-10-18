import sympy as sp

from ..operand import Operand
from ..lambda_ring_context import LambdaRingContext

def get_max_adams_degree_nary(self: sp.Add | sp.Mul) -> int:
    """
    Computes the maximum Adams degree of this expression tree.

    This function calculates the maximum Adams degree by traversing the expression tree,
    specifically for `Add` and `Mul` expressions. It multiplies all of the Adams, lambda, 
    and sigma degrees of each branch in the tree and returns the maximum value.

    This function is intended to be added as a method for `sp.Add` and `sp.Mul`.

    Returns:
    --------
    int
        The maximum Adams degree of this tree after conversion to an Adams polynomial.
    """
    return max(child.get_max_adams_degree() for child in self.args)


def get_max_groth_degree_nary(self: sp.Add | sp.Mul) -> int:
    """
    Computes the maximum degree required to create a Grothendieck context for this expression tree.

    This function calculates the maximum sigma or lambda degree by traversing the branches 
    of the `Add` or `Mul` expressions. It finds the highest degree required for Grothendieck 
    context creation.

    This function is intended to be added as a method for `sp.Add` and `sp.Mul`.

    Returns:
    --------
    int
        The maximum sigma or lambda degree required for the tree.
    """
    return max(child.get_max_groth_degree() for child in self.args)


def _to_adams_nary(
    self: sp.Add | sp.Mul,
    operands: set[Operand],
    lrc: LambdaRingContext,
) -> sp.Expr:
    """
    Converts this subtree into an equivalent Adams polynomial.

    This function converts `Add` and `Mul` expressions into their corresponding Adams 
    polynomial by recursively converting each child node. Each child of the `Add` or 
    `Mul` expression is transformed using the `_to_adams` method, and the resulting 
    Adams polynomials are combined.

    This function is intended to be added as a method for `sp.Add` and `sp.Mul`.

    Args:
    -----
    operands : set[Operand]
        The set of all operands in the expression tree.
    lrc : LambdaRingContext
        The Grothendieck ring context used for the conversion between ring operators.

    Returns:
    --------
    sp.Expr
        A polynomial of Adams operators equivalent to this subtree.
    """
    return type(self)(*(child._to_adams(operands, lrc) for child in self.args))


def _to_adams_lambda_nary(
    self: sp.Add | sp.Mul,
    operands: set[Operand],
    lrc: LambdaRingContext,
    adams_degree: int = 1,
) -> sp.Expr:
    """
    Converts this subtree into an equivalent Adams polynomial, optimized for lambda conversion.

    This function is similar to `_to_adams_nary`, but it is optimized for lambda 
    conversions when called from `to_lambda`. Each child of the `Add` or `Mul` expression 
    is recursively converted using the `_to_adams_lambda` method.

    This function is intended to be added as a method for `sp.Add` and `sp.Mul`.

    Args:
    -----
    operands : set[Operand]
        The set of all operands in the expression tree.
    lrc : LambdaRingContext
        The Grothendieck ring context used for the conversion between ring operators.
    adams_degree : int, optional
        The cumulative Adams degree higher than this node in the expression tree.

    Returns:
    --------
    sp.Expr
        A polynomial of Adams operators equivalent to this subtree.
    """
    return type(self)(
        *(child._to_adams_lambda(operands, lrc, adams_degree) for child in self.args)
    )