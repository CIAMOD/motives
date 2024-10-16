# operand.py

from __future__ import annotations  # For forward references
import sympy as sp
from multipledispatch import dispatch

from .core.operand import Operand
from .core.lambda_ring_context import LambdaRingContext

class Variable(Operand, sp.Symbol):
    """
    Represents an abstract variable node in an expression.

    A `Variable` inherits from both `Operand` and SymPy's `Symbol`, and it is used to represent
    a variable in an expression tree. This class cannot be named 'L' (reserved for the Lefschetz 
    motive) or start with 's_' (reserved for special symbols), to prevent naming conflicts with 
    SymPy symbols.

    Attributes:
    -----------
    name : str
        A string representing the name of the variable.
    _adams_vars : list[sp.AtomicExpr]
        A list of Adams variables generated for this variable, with degree-based indexing.
    _lambda_vars : list[sp.AtomicExpr]
        A list of Lambda variables generated for this variable, with degree-based indexing.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes a `Variable` with empty lists of Adams and Lambda variables.

        Args:
        -----
        *args : tuple
            Positional arguments passed to the superclass constructor.
        **kwargs : dict
            Keyword arguments passed to the superclass constructor.
        """
        self._adams_vars: list[sp.AtomicExpr] = [sp.Integer(1), self]
        self._lambda_vars: list[sp.AtomicExpr] = [sp.Integer(1), self]

    def __new__(cls, name: str, **assumptions):
        """
        Creates a new instance of `Variable` while enforcing naming restrictions.

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
        Variable
            A new instance of `Variable`.
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

    def _generate_adams_vars(self, n: int) -> None:
        """
        Generates the Adams variables for this variable up to degree `n`.

        Args:
        -----
        n : int
            The maximum degree of Adams needed.
        """
        self._adams_vars += [
            sp.Symbol(f"ψ_{i}({self})") for i in range(len(self._adams_vars), n + 1)
        ]

    def _generate_lambda_vars(self, n: int) -> None:
        """
        Generates the Lambda variables for this variable up to degree `n`.

        Args:
        -----
        n : int
            The maximum degree of Lambda needed.
        """
        self._lambda_vars += [
            sp.Symbol(f"λ_{i}({self})") for i in range(len(self._lambda_vars), n + 1)
        ]

    def get_adams_var(self, i: int) -> sp.Expr:
        """
        Returns the Adams variable of this variable for a given degree `i`.

        Args:
        -----
        i : int
            The degree of the Adams operator.

        Returns:
        --------
        sp.Expr
            The Adams variable for degree `i`.
        """
        self._generate_adams_vars(i)
        return self._adams_vars[i]

    def get_lambda_var(self, i: int, context: LambdaRingContext = None) -> sp.Expr:
        """
        Returns the Lambda variable of this variable for a given degree `i`.

        Args:
        -----
        i : int
            The degree of the Lambda operator.
        context : LambdaRingContext, optional
            The ring context used for the conversion between operators.

        Returns:
        --------
        sp.Expr
            The Lambda variable for degree `i`.
        """
        self._generate_lambda_vars(i)
        return self._lambda_vars[i]

    @dispatch(int, sp.Expr)
    def _to_adams(self, degree: int, ph: sp.Expr) -> sp.Expr:
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

        Returns:
        --------
        sp.Expr
            The polynomial with the Adams operator applied.
        """
        max_adams = -1
        operands = ph.free_symbols
        for i, adams in reversed(list(enumerate(self._adams_vars))):
            if adams in operands:
                max_adams = i
                break

        return ph.xreplace(
            {
                self.get_adams_var(i): self.get_adams_var(degree * i)
                for i in range(1, max_adams + 1)
            }
        )

    @dispatch(set, LambdaRingContext)
    def _to_adams(self, operands: set[Operand], lrc: LambdaRingContext) -> sp.Expr:
        """
        Converts this variable into an equivalent Adams polynomial.

        Args:
        -----
        operands : set[Operand]
            The set of all operands in the expression tree.
        lrc : LambdaRingContext
            The Grothendieck ring context used for the conversion between ring operators.

        Returns:
        --------
        sp.Expr
            The Adams polynomial for this variable.
        """
        return self.get_adams_var(1)

    def _subs_adams(self, lrc: LambdaRingContext, ph: sp.Expr) -> sp.Expr:
        """
        Substitutes Adams variables of this variable in a polynomial with Lambda polynomials.

        This method is used to convert Adams polynomials into Lambda polynomials.

        Args:
        -----
        lrc : LambdaRingContext
            The Grothendieck ring context used for the conversion between ring operators.
        ph : sp.Expr
            The polynomial in which to substitute the Adams variables.

        Returns:
        --------
        sp.Expr
            The polynomial with Adams variables replaced by Lambda variables.
        """
        ph = ph.xreplace(
            {
                self.get_adams_var(i): lrc.get_lambda_2_adams_pol(i)
                for i in range(2, len(self._adams_vars))
            }
        )
        ph = ph.xreplace(
            {
                lrc.lambda_vars[i]: self.get_lambda_var(i)
                for i in range(1, len(lrc.lambda_vars))
            }
        )
        return ph
