# operand.py

from __future__ import annotations  # For forward references
import sympy as sp
import re

from .core.operand.operand import Operand
from .core.operator.ring_operator import Adams, Lambda_
from .core.lambda_ring_context import LambdaRingContext
from .utils import preorder_traversal


class Free(Operand, sp.Symbol):
    """
    Represents an abstract variable node in an expression, treated as a generator of a
    free lambda-ring or a free extension of a lambda-ring.

    The lambda operations of a Free object are considered by default as algebraically independent
    elements in the expression. New SymPy Symbols are created for each necessary lambda operation and
    Adams operation of a Free object when they appear in an expression.

    A `Free` inherits from both `Operand` and SymPy's `Symbol`, and it is used to represent
    a variable in an expression tree. This class cannot be named 'L' (reserved for the Lefschetz
    motive) or start with 's_' (reserved for special symbols), to prevent naming conflicts with
    SymPy symbols.

    Attributes:
    -----------
    name : str
        A string representing the name of the variable.
    """

    adams_pattern = re.compile(r"ψ(\d+)\((\w+)\)")

    def __new__(cls, name: str, **assumptions):
        """
        Creates a new instance of a "Free" lambda-ring variable while enforcing naming restrictions.

        Args:
        -----
        name : str
            The name of the variable.
        assumptions : dict
            Assumptions passed to the SymPy `Symbol` constructor.

        Raises:
        -------
        ValueError
            If the name is 'L' (reserved for the Lefschetz motive) or starts with 's_'.

        Returns:
        --------
        Free
            A new instance of `Free`.
        """
        if name == "L":
            raise ValueError("The name 'L' is reserved for the Lefschetz motive.")
        if name[0] == "s" and name[1] == "_":
            raise ValueError("The name 's_[name]' is reserved for the symbol.")

        return sp.Symbol.__new__(cls, name, **assumptions)

    def __repr__(self) -> str:
        """
        Returns the string representation of the variable, which is its name.

        Returns:
        --------
        str
            The name of the variable.
        """
        return self.name

    def get_adams_var(self, i: int, as_symbol: bool = False) -> sp.Expr:
        """
        Returns the Adams variable of this variable for a given degree `i`.

        Args:
        -----
        i : int
            The degree of the Adams operator.
        as_symbol : bool, optional
            If True, returns the Adams variable as a SymPy Symbol. Otherwise, returns it as an
            Adams object.

        Returns:
        --------
        sp.Expr
            The Adams variable for degree `i`.
        """
        if i == 0:
            return 1
        if i == 1:
            return self
        if as_symbol is False:
            return self.adams(i)
        return sp.Symbol(f"ψ{i}({self})")

    def get_lambda_var(self, i: int, as_symbol: bool = False) -> sp.Expr:
        """
        Returns the lambda variable of this variable for a given degree `i`.

        Args:
        -----
        i : int
            The degree of the lambda operator.
        as_symbol : bool, optional
            If True, returns the lambda variable as a SymPy Symbol. Otherwise, returns it as a
            Lambda_ object.

        Returns:
        --------
        sp.Expr
            The lambda variable for degree `i`.
        """
        if as_symbol is False:
            return self.lambda_(i)
        return sp.Symbol(f"λ{i}({self})")

    def _apply_adams(
        self, degree: int, ph: sp.Expr, max_adams_degree: int, as_symbol: bool = False
    ) -> sp.Expr:
        """
        Applies the Adams operator to all instances of this variable in a polynomial.

        It replaces all instances of an Adams variable (ψ_d(variable)) in the polynomial
        for ψ_{degree*d}(variable).

        Args:
        -----
        degree : int
            The degree of the Adams operator to apply.
        ph : sp.Expr
            The polynomial in which to apply the Adams operator.
        max_adams_degree : int
            The maximum Adams degree in the expression.
        as_symbol : bool, optional
            If True, ph is a polynomial in SymPy symbols. Otherwise, ph is a polynomial in
            Adams_ objects.

        Returns:
        --------
        sp.Expr
            The polynomial with the Adams operator applied.
        """
        return ph.xreplace(
            {
                self.get_adams_var(i, as_symbol=as_symbol): self.get_adams_var(
                    i * degree, as_symbol=as_symbol
                )
                for i in range(1, max_adams_degree + 1)
            }
        )

    def _subs_adams(
        self, ph: sp.Expr, max_adams_degree: int, as_symbol: bool = False
    ) -> sp.Expr:
        """
        Given a polynomial ph, substitutes all the appearences of Adams variables of this variable
        in ph by the corresponding polynomials in lambda operations of this variable representing them.

        This method is used to convert Adams polynomials into Lambda polynomials.

        Args:
        -----
        ph : sp.Expr
            The polynomial in which to substitute the Adams variables.
        max_adams_degree : int
            The maximum Adams degree in the expression.
        as_symbol : bool, optional
            If True, ph is a polynomial in SymPy symbols. Otherwise, ph is a polynomial in
            Adams_ objects.

        Returns:
        --------
        sp.Expr
            The polynomial with Adams variables of this variable replaced by polynomials in lambda
            variables of this variable.
        """
        lrc = LambdaRingContext()

        ph = ph.xreplace(
            {
                self.get_adams_var(i, as_symbol=as_symbol): lrc.get_lambda_2_adams_pol(
                    i
                )
                for i in range(1, max_adams_degree + 1)
            }
        )
        ph = ph.xreplace(
            {
                lrc.lambda_vars[i]: self.get_lambda_var(i, as_symbol=as_symbol)
                for i in range(1, len(lrc.lambda_vars))
            }
        )
        return ph
