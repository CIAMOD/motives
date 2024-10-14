from __future__ import annotations
from multipledispatch import dispatch
import sympy as sp

from ..utils import SingletonMeta

from ..core import GrothendieckRingContext
from ..core.operand import Operand

from .motive import Motive

class Point(Motive, sp.AtomicExpr, metaclass=SingletonMeta):
    """
    Represents a point motive in an expression.

    The point motive is a universal motive and acts as a singleton. It inherits from 
    `Motive` and SymPy's `AtomicExpr`, meaning it is treated as an indivisible expression 
    and supports Adams and Lambda operations, though they always return 1 for this motive.
    """

    def __new__(cls) -> Point:
        """
        Creates a new instance of `Point`, enforcing singleton behavior.

        Returns:
        --------
        Point
            The singleton instance of the `Point` class.
        """
        instance = sp.AtomicExpr.__new__(cls)
        instance._assumptions["commutative"] = True
        return instance

    def __repr__(self) -> str:
        """
        Returns the string representation of the point motive.

        Returns:
        --------
        str
            The string "pt", representing the point motive.
        """
        return "pt"

    def get_adams_var(self, i: int) -> sp.Expr:
        """
        Returns the point motive with an Adams operation applied to it.

        Since the Adams operation on a point is always 1, this method always returns 1, 
        regardless of the degree `i`.

        Args:
        -----
        i : int
            The degree of the Adams operator.

        Returns:
        --------
        sp.Expr
            The result of applying the Adams operator to the point, which is always 1.
        """
        return sp.Integer(1)

    def get_lambda_var(self, i: int, context: GrothendieckRingContext = None) -> sp.Expr:
        """
        Returns the point motive with a Lambda operation applied to it.

        Since the Lambda operation on a point is always 1, this method always returns 1, 
        regardless of the degree `i`.

        Args:
        -----
        i : int
            The degree of the Lambda operator.
        context : GrothendieckRingContext, optional
            The ring context used for the conversion between operators (not used here).

        Returns:
        --------
        sp.Expr
            The result of applying the Lambda operator to the point, which is always 1.
        """
        return sp.Integer(1)

    @dispatch(int, object)
    def _to_adams(self, degree: int, ph: sp.Expr) -> sp.Expr:
        """
        Applies the Adams operator to instances of the point in a polynomial.

        Since the Adams operation on a point is always 1, this method returns the polynomial 
        `ph` unchanged.

        Args:
        -----
        degree : int
            The degree of the Adams operator to apply.
        ph : sp.Expr
            The polynomial in which the Adams operator is applied.

        Returns:
        --------
        sp.Expr
            The original polynomial, unchanged.
        """
        return ph

    @dispatch(set, GrothendieckRingContext)
    def _to_adams(self, operands: set[Operand], gc: GrothendieckRingContext) -> sp.Expr:
        """
        Converts this point into an equivalent Adams polynomial.

        Since the Adams operation on a point is always 1, this method always returns 1.

        Args:
        -----
        operands : set[Operand]
            The set of all operands in the expression tree.
        gc : GrothendieckRingContext
            The Grothendieck ring context used for the conversion between ring operators.

        Returns:
        --------
        sp.Expr
            The Adams polynomial of the point, which is always 1.
        """
        return sp.Integer(1)

    def _subs_adams(self, gc: GrothendieckRingContext, ph: sp.Expr) -> sp.Expr:
        """
        Substitutes Adams variables in a polynomial with their equivalent Lambda polynomials.

        Since no Adams variables are generated for a point, this method returns the polynomial 
        `ph` unchanged. This method is called in `to_lambda` to substitute any Adams variables 
        after converting the expression tree to an Adams polynomial.

        Args:
        -----
        gc : GrothendieckRingContext
            The Grothendieck ring context used for the conversion between ring operators.
        ph : sp.Expr
            The polynomial in which to substitute the Adams variables.

        Returns:
        --------
        sp.Expr
            The original polynomial, unchanged.
        """
        return ph
