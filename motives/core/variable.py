# operand.py

from __future__ import annotations  # For forward references
from typing import Optional, Hashable, Set
import sympy as sp
from multipledispatch import dispatch

from .node import Node
from .operand import Operand
from .lambda_context import LambdaContext

class Variable(Operand):
    """
    An operand node in an expression tree that represents an abstract variable.

    Parameters:
    -----------
    value : sympy.Symbol
        The value of the variable, i.e. how it is represented.
    parent : Node
        The parent node of the variable. If the variable is the root
        of the tree, parent is None.
    id_ : hashable
        The id of the variable, used to identify it (if two operands
        have the same id, they are the same operand).

    Methods:
    --------
    sigma(degree: int) -> ET
        Applies the sigma operation to the current variable.
    lambda_(degree: int) -> ET
        Applies the lambda operation to the current variable.
    adams(degree: int) -> ET
        Applies the adams operation to the current variable.

    Properties:
    -----------
    sympy : sp.Expr
        The sympy representation of the node (and any children).
    """

    def __init__(
        self,
        value: sp.Symbol | str,
        parent: Optional[Node] = None,
        id_: Hashable = None,
    ):
        if isinstance(value, str):
            value = sp.Symbol(value)
        self.value: sp.Symbol = value

        super().__init__(parent, id_)
        self._adams_vars: list[sp.Symbol] = [1, self.value]
        self._lambda_vars: list[sp.Symbol] = [1, self.value]
        self._generate_adams_vars(1)

    def __repr__(self) -> str:
        return str(self.value)

    def _generate_adams_vars(self, n: int) -> None:
        """
        Generate the adams variables needed up to degree n.
        """
        self._adams_vars += [
            sp.Symbol(f"ψ_{i}({self.value})")
            for i in range(len(self._adams_vars), n + 1)
        ]

    def _generate_lambda_vars(self, n: int) -> None:
        """
        Generate the lambda variables needed up to degree n.
        """
        self._lambda_vars += [
            sp.Symbol(f"λ_{i}({self.value})")
            for i in range(len(self._lambda_vars), n + 1)
        ]

    def get_adams_var(self, i: int) -> sp.Symbol:
        """
        Returns the adams variable of degree i.

        Parameters
        ----------
        i : int
            The degree of the adams variable.

        Returns
        -------
        The adams variable of degree i.
        """
        self._generate_adams_vars(i)
        return self._adams_vars[i]

    def get_lambda_var(self, i: int) -> sp.Symbol:
        """
        Returns the lambda variable of degree i.

        Parameters
        ----------
        i : int
            The degree of the lambda variable.

        Returns
        -------
        The lambda variable of degree i.
        """
        self._generate_lambda_vars(i)
        return self._lambda_vars[i]

    @dispatch(int, int)
    def _to_adams(self, degree: int, ph: int) -> int:
        """
        Catches the case where the polynomial is an integer.
        """
        return ph

    @dispatch(int, sp.Expr)
    def _to_adams(self, degree: int, ph: sp.Expr) -> sp.Expr:
        """
        Applies an adams operation of degree `degree` to any instances of this variable
        in the polynomial `ph`.
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

    @dispatch(set, LambdaContext)
    def _to_adams(
        self, operands: set[Operand], group_context: LambdaContext
    ) -> sp.Expr:
        """
        Returns the adams of degree 1 of the variable.
        """
        return self.get_adams_var(1)

    def _subs_adams(self, group_context: LambdaContext, ph: sp.Expr) -> sp.Expr:
        """
        Substitutes any instances of an adams of this variable (ψ_d(variable)) into its
        equivalent polynomial of lambdas.
        """
        ph = ph.xreplace(
            {
                self.get_adams_var(i): group_context.get_lambda_2_adams_pol(i)
                for i in range(2, len(self._adams_vars))
            }
        )
        ph = ph.xreplace(
            {
                group_context.lambda_vars[i]: self.get_lambda_var(i)
                for i in range(1, len(group_context.lambda_vars))
            }
        )
        return ph

    @property
    def sympy(self) -> sp.Symbol:
        """
        The sympy representation of the variable.
        """
        return self.value