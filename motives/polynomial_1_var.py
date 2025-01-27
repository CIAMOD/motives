from __future__ import annotations
import sympy as sp
import warnings

from .core.operand.operand import Operand
from .core.operand.object_1_dim import Object1Dim


class Polynomial1Var(Object1Dim, sp.Symbol):
    """
    Represents an abstract one-dimensional motive in an expression tree.

    A `Polynomial1Var` is an operand that can be used in expressions and supports Adams and Lambda operations,
    which are equivalent to raising the Polynomial1Var to the specified power.

    Attributes:
    -----------
    name : str
        The name of the Polynomial1Var.
    """

    def __new__(cls, name: str, **assumptions) -> Polynomial1Var:
        """
        Creates a new instance of `Polynomial1Var`, with special handling for the name 'L',
        which is reserved for the Lefschetz motive.

        If the name 'L' is used, a warning is issued, and the Polynomial1Var is replaced by
        the Lefschetz motive.

        Args:
        -----
        name : str
            The name of the Polynomial1Var.
        assumptions : dict
            Additional assumptions passed to the `Polynomial1Var` constructor.

        Returns:
        --------
        Polynomial1Var or Lefschetz
            A new instance of `Polynomial1Var` or the `Lefschetz` motive if the name is 'L'.
        """
        if name == "L":
            warnings.warn(
                "The name 'L' is reserved for the Lefschetz motive. Using 'Lefschetz' instead."
            )
            from .grothendieck_motives.lefschetz import Lefschetz

            return Lefschetz()
        return sp.Symbol.__new__(cls, f"s_{name}", **assumptions)

    def __repr__(self) -> str:
        """
        Returns the string representation of the Polynomial1Var, which is its name.

        Returns:
        --------
        str
            The string representation of the Polynomial1Var.
        """
        return self.name

    def get_adams_var(self, i: int, as_symbol: bool = False) -> sp.Expr:
        """
        Returns the Polynomial1Var with an Adams operation applied to it.

        The Adams operation on a Polynomial1Var is equivalent to raising the Polynomial1Var to the i-th power.

        Args:
        -----
        i : int
            The degree of the Adams operator.
        as_symbol : bool, optional
            If True, returns the Adams variable as a SymPy Symbol. Otherwise, returns it as an
            Adams_ object.

        Returns:
        --------
        sp.Expr
            The Polynomial1Var raised to the i-th power.
        """
        return self**i

    def get_lambda_var(self, i: int, as_symbol: bool = False) -> sp.Expr:
        """
        Returns the Polynomial1Var with a Lambda operation applied to it.

        The Lambda operation on a Polynomial1Var is equivalent to raising the Polynomial1Var to the i-th power.

        Args:
        -----
        i : int
            The degree of the Lambda operator.
        as_symbol : bool, optional
            If True, returns the lambda variable as a SymPy Symbol. Otherwise, returns it as a
            Lambda_ object.

        Returns:
        --------
        sp.Expr
            The Polynomial1Var raised to the i-th power.
        """
        return self**i

    def _apply_adams(
        self, degree: int, ph: sp.Expr, max_adams_degree: int, as_symbol: bool = False
    ) -> sp.Expr:
        """
        Applies the Adams operator to any instances of this Polynomial1Var in a polynomial.

        It replaces all instances of the Polynomial1Var in the polynomial with the Polynomial1Var raised
        to the specified power.

        Args:
        -----
        degree : int
            The degree of the Adams operator to apply.
        ph : sp.Expr
            The polynomial in which the Adams operator is applied.
        max_adams_degree : int
            The maximum Adams degree in the expression.
        as_symbol : bool, optional
            If True, returns the Adams variable as a SymPy Symbol. Otherwise, returns it as an
            Adams object.

        Returns:
        --------
        sp.Expr
            The polynomial with the Adams operator applied to the Polynomial1Var.
        """
        return ph.xreplace({self: self.get_adams_var(degree, as_symbol=as_symbol)})

    def _subs_adams(
        self, ph: sp.Expr, max_adams_degree: int, as_symbol: bool = False
    ) -> sp.Expr:
        """
        Substitutes Adams variables of this Polynomial1Var in a polynomial with their equivalent Lambda polynomials.

        Since no specific Adams variables are generated for a Polynomial1Var, this method returns the polynomial
        unchanged. It is called during the `to_lambda` process to substitute any Adams variables in
        the polynomial after converting the expression tree to an Adams polynomial.

        Args:
        -----
        ph : sp.Expr
            The polynomial in which to substitute the Adams variables.
        max_adams_degree : int
            The maximum Adams degree in the expression.
        as_symbol : bool, optional
            If True, the polynomial is a polynomial in SymPy symbols. Otherwise, the polynomial is a
            polynomial in Adams objects.

        Returns:
        --------
        sp.Expr
            The original polynomial, unchanged.
        """
        return ph
