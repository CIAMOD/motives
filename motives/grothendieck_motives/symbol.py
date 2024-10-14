from __future__ import annotations
from multipledispatch import dispatch
import sympy as sp
import warnings

from ..core import GrothendieckRingContext
from ..core.operand import Operand

from .motive import Motive

class Symbol(Motive, sp.Symbol):
    """
    Represents an abstract one-dimensional motive in an expression tree.

    A `Symbol` is an operand that can be used in expressions and supports Adams and Lambda operations, 
    which are equivalent to raising the symbol to the specified power.

    Attributes:
    -----------
    name : str
        The name of the symbol.
    """

    def __new__(cls, name: str, **assumptions) -> Symbol:
        """
        Creates a new instance of `Symbol`, with special handling for the name 'L', 
        which is reserved for the Lefschetz motive.

        If the name 'L' is used, a warning is issued, and the symbol is replaced by 
        the Lefschetz motive.

        Args:
        -----
        name : str
            The name of the symbol.
        assumptions : dict
            Additional assumptions passed to the `Symbol` constructor.

        Returns:
        --------
        Symbol or Lefschetz
            A new instance of `Symbol` or the `Lefschetz` motive if the name is 'L'.
        """
        if name == "L":
            warnings.warn(
                "The name 'L' is reserved for the Lefschetz motive. Using 'Lefschetz' instead."
            )
            from .lefschetz import Lefschetz
            return Lefschetz()
        return sp.Symbol.__new__(cls, f"s_{name}", **assumptions)

    def __repr__(self) -> str:
        """
        Returns the string representation of the symbol, which is its name.

        Returns:
        --------
        str
            The string representation of the symbol.
        """
        return self.name

    def get_adams_var(self, i: int) -> sp.Expr:
        """
        Returns the symbol with an Adams operation applied to it.

        The Adams operation on a symbol is equivalent to raising the symbol to the i-th power.

        Args:
        -----
        i : int
            The degree of the Adams operator.

        Returns:
        --------
        sp.Expr
            The symbol raised to the i-th power.
        """
        return self**i

    def get_lambda_var(self, i: int, context: GrothendieckRingContext = None) -> sp.Expr:
        """
        Returns the symbol with a Lambda operation applied to it.

        The Lambda operation on a symbol is equivalent to raising the symbol to the i-th power.

        Args:
        -----
        i : int
            The degree of the Lambda operator.
        context : GrothendieckRingContext, optional
            The ring context used for the conversion between operators.

        Returns:
        --------
        sp.Expr
            The symbol raised to the i-th power.
        """
        return self**i

    @dispatch(int, sp.Expr)
    def _to_adams(self, degree: int, ph: sp.Expr) -> sp.Expr:
        """
        Applies the Adams operator to any instances of this symbol in a polynomial.

        It replaces all instances of the symbol in the polynomial with the symbol raised 
        to the specified power.

        Args:
        -----
        degree : int
            The degree of the Adams operator to apply.
        ph : sp.Expr
            The polynomial in which the Adams operator is applied.

        Returns:
        --------
        sp.Expr
            The polynomial with the Adams operator applied to the symbol.
        """
        return ph.xreplace({self: self.get_adams_var(degree)})

    @dispatch(set, GrothendieckRingContext)
    def _to_adams(self, operands: set[Operand], gc: GrothendieckRingContext) -> sp.Expr:
        """
        Converts this symbol into an equivalent Adams polynomial.

        For a symbol, this simply returns the symbol itself.

        Args:
        -----
        operands : set[Operand]
            The set of all operands in the expression tree.
        gc : GrothendieckRingContext
            The Grothendieck ring context used for the conversion between ring operators.

        Returns:
        --------
        sp.Expr
            The symbol itself.
        """
        return self

    def _subs_adams(self, gc: GrothendieckRingContext, ph: sp.Expr) -> sp.Expr:
        """
        Substitutes Adams variables of this symbol in a polynomial with their equivalent Lambda polynomials.

        Since no specific Adams variables are generated for a symbol, this method returns the polynomial 
        unchanged. It is called during the `to_lambda` process to substitute any Adams variables in 
        the polynomial after converting the expression tree to an Adams polynomial.

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
