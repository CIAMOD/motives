import sympy as sp
from typeguard import typechecked

from . import Adams, Lambda_, Sigma
from ..groth_ring_context import GrothendieckRingContext
from ..operand import Operand

def to_adams(self: sp.Expr, gc: GrothendieckRingContext = None) -> sp.Expr:
    """
    Converts the current expression into a polynomial of Adams operators.

    This function transforms the expression into its equivalent form using Adams operators. 
    It is intended to be used as a method for `Pow`, `Add`, `Mul`, and `Rational` expressions. 
    The Adams polynomial represents the structure of the current expression in the context of 
    Grothendieck rings.

    Args:
    -----
    gc : GrothendieckRingContext, optional
        The Grothendieck ring context to use for the conversion. If not provided, a new context is created.

    Returns:
    --------
    sp.Expr
        The polynomial of Adams operators equivalent to the current expression.
    """
    gc = gc or GrothendieckRingContext()

    operands: set[Operand] = self.free_symbols

    return self._to_adams(operands, gc)

def to_lambda(self: sp.Expr, gc: GrothendieckRingContext = None, *, optimize=True) -> sp.Expr:
    """
    Converts the current expression into a polynomial of lambda operators.

    This function transforms the expression into its equivalent form using lambda operators.
    It is intended to be used as a method for `Pow`, `Add`, `Mul`, and `Rational` expressions. 
    The conversion is done via Adams operators, which are then substituted by their lambda equivalents.
    
    Args:
    -----
    gc : GrothendieckRingContext, optional
        The Grothendieck ring context of the tree. If not provided, a new context is created.
    optimize : bool, optional, default=True
        Whether to optimize the conversion to lambda. If True, the conversion uses an optimized 
        pathway by converting Adams operators directly to lambda operators.

    Returns:
    --------
    sp.Expr
        The polynomial of lambda operators equivalent to the current expression.
    """
    # Initialize a Grothendieck ring context if not provided
    gc = gc or GrothendieckRingContext()

    operands: set[Operand] = self.free_symbols

    # Get the Adams polynomial of the tree, with optimization if requested
    if optimize:
        adams_pol = self._to_adams_lambda(operands, gc, 1)
    else:
        adams_pol = self._to_adams(operands, gc)

    # If the result is an integer, return it directly
    if isinstance(adams_pol, (sp.Integer, int)):
        return adams_pol

    # Substitute Adams variables for lambda variables
    for operand in operands:
        adams_pol = operand._subs_adams(gc, adams_pol)

    return adams_pol

@typechecked
def sigma(self: sp.Expr, degree: int) -> sp.Expr:
    """
    Applies the sigma operation to the current expression.

    This function adds the sigma ring operator to the expression, creating a new expression
    with the specified sigma operator applied. It is intended to be used as a method for 
    `Pow`, `Add`, `Mul`, and `Rational` expressions.

    Args:
    -----
    degree : int
        The degree of the sigma operator to be applied.

    Returns:
    --------
    sp.Expr
        An expression with the sigma operator applied.
    """
    return Sigma(degree, self)


@typechecked
def lambda_(self: sp.Expr, degree: int) -> sp.Expr:
    """
    Applies the lambda operation to the current expression.

    This function adds the lambda ring operator to the expression, creating a new expression
    with the specified lambda operator applied. It is intended to be used as a method for 
    `Pow`, `Add`, `Mul`, and `Rational` expressions.

    Args:
    -----
    degree : int
        The degree of the lambda operator to be applied.

    Returns:
    --------
    sp.Expr
        An expression with the lambda operator applied.
    """
    return Lambda_(degree, self)


@typechecked
def adams(self: sp.Expr, degree: int) -> sp.Expr:
    """
    Applies the Adams operation to the current expression.

    This function adds the Adams ring operator to the expression, creating a new expression
    with the specified Adams operator applied. It is intended to be used as a method for 
    `Pow`, `Add`, `Mul`, and `Rational` expressions.

    Args:
    -----
    degree : int
        The degree of the Adams operator to be applied.

    Returns:
    --------
    sp.Expr
        An expression with the Adams operator applied.
    """
    return Adams(degree, self)
