# operand.py

from __future__ import annotations  # For forward references
from typing import Optional, Hashable
import sympy as sp
from math import comb
from multipledispatch import dispatch

from .core.lambda_ring_expr import LambdaRingExpr
from .core.operand import Operand
from .core.lambda_ring_context import LambdaRingContext

class Integer(Operand):
    """
    An operand in an expression tree that represents an integer.

    Parameters:
    -----------
    value : int
        The value of the integer.
    parent : LambdaRingExpr
        The parent node of the integer.
    id_ : hashable
        The id of the integer, used to identify it (if two operands
        have the same id, they are the same operand).

    Methods:
    --------
    sigma(degree: int) -> ET
        Applies the sigma operation to the current integer.
    lambda_(degree: int) -> ET
        Applies the lambda operation to the current integer.
    adams(degree: int) -> ET
        Applies the adams operation to the current integer.

    Properties:
    -----------
    sympy : sp.Integer
        The sympy representation of the integer.
    """

    def __init__(
        self, value: int = 1, parent: Optional[LambdaRingExpr] = None, id_: Hashable = None
    ):
        self.value: int = value
        super().__init__(parent, id_)

    def __repr__(self) -> str:
        return str(self.value)

    def get_adams_var(self, i: int) -> int:
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
        return self.value

    def get_lambda_var(self, i: int) -> int:
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
        return comb(i + self.value - 1, i)

    @dispatch(int, object)
    def _to_adams(self, degree: int, ph: sp.Expr) -> sp.Expr:
        """
        Applies an adams operation of degree `degree` to any instances of this integer
        in the polynomial `ph`.
        """
        return ph

    @dispatch(set, LambdaRingContext)
    def _to_adams(
        self, operands: set[Operand], group_context: LambdaRingContext
    ) -> sp.Expr:
        """
        Returns the integer.
        """
        return sp.Integer(self.value)

    def _subs_adams(self, group_context: LambdaRingContext, ph: sp.Expr) -> sp.Expr:
        """
        Substitutes any instances of an adams of this integer into its equivalent
        polynomial of lambdas.
        """
        return ph

    @property
    def sympy(self) -> sp.Integer:
        """
        The sympy representation of the integer.
        """
        return sp.Integer(self.value)