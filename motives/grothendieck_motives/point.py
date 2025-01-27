from __future__ import annotations
import sympy as sp

from ..utils import SingletonMeta

from .motive import Motive


class Point(Motive, sp.AtomicExpr, metaclass=SingletonMeta):
    """
    Represents the motive of a closed point in the Grothendieck lambda-ring of
    varieties, the Grothendieck ring of Chow motives, or any extension or completion
    of such rings.

    The point motive is a universal motive and acts as a singleton. It inherits from
    `Motive` and SymPy's `AtomicExpr`, making it an indivisible expression that supports
    Adams and lambda operations, which always return 1 for this motive.
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

    def get_adams_var(self, i: int, as_symbol: bool = False) -> sp.Expr:
        """
        Returns the point motive with an Adams operation applied.

        Since the Adams operation on a point is always 1, this method always returns 1.

        Parameters:
        -----------
        i : int
            The degree of the Adams operator.
        as_symbol : bool, optional
            If True, returns the Adams variable as a SymPy Symbol. Otherwise, returns 1.

        Returns:
        --------
        sp.Expr
            The result of applying the Adams operator to the point, which is always 1.
        """
        return sp.Integer(1)

    def get_lambda_var(self, i: int, as_symbol: bool = False) -> sp.Expr:
        """
        Returns the point motive with a Lambda operation applied.

        Since the Lambda operation on a point is always 1, this method always returns 1.

        Parameters:
        -----------
        i : int
            The degree of the Lambda operator.
        as_symbol : bool, optional
            If True, returns the Lambda variable as a SymPy Symbol. Otherwise, returns 1.

        Returns:
        --------
        sp.Expr
            The result of applying the Lambda operator to the point, which is always 1.
        """
        return sp.Integer(1)

    def _apply_adams(
        self, degree: int, ph: sp.Expr, max_adams_degree: int, as_symbol: bool = False
    ) -> sp.Expr:
        """
        Applies the Adams operator to instances of the point in a polynomial.

        Since the Adams operation on a point is always 1, this method returns the polynomial unchanged.

        Parameters:
        -----------
        degree : int
            The degree of the Adams operator to apply.
        ph : sp.Expr
            The polynomial in which the Adams operator is applied.
        max_adams_degree : int
            The maximum degree of Adams operators in the expression.
        as_symbol : bool, optional
            If True, represents Adams operators as symbols.

        Returns:
        --------
        sp.Expr
            The original polynomial, unchanged.
        """
        return ph

    def _subs_adams(
        self, ph: sp.Expr, max_adams_degree: int, as_symbol: bool = False
    ) -> sp.Expr:
        """
        Substitutes Adams variables in a polynomial with their equivalent Lambda polynomials.

        Since no Adams variables are generated for a point, this method returns the polynomial unchanged.

        Parameters:
        -----------
        ph : sp.Expr
            The polynomial in which to substitute the Adams variables.
        max_adams_degree : int
            The maximum degree of Adams operators in the expression.
        as_symbol : bool, optional
            If True, represents Lambda operators as symbols.

        Returns:
        --------
        sp.Expr
            The original polynomial, unchanged.
        """
        return ph
