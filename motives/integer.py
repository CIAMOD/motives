# operand.py

from __future__ import annotations  # For forward references
from typing import Optional, Hashable
import sympy as sp
from math import comb
from multipledispatch import dispatch

from .core.lambda_ring_expr import LambdaRingExpr
from .core.operand import Operand


class Integer(Operand):
    """
    An operand in an expression tree that represents an integer and such the
    lambda, sigma and Adams operations act on this element following the natural lambda-ring
    structure on Z, namely
        λ^n(x)=binom(n+x-1,n)
        σ^n(x)=binom(x,n)
        psi^n(x)=x

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
        Applies the Adams operation to the current integer.

    Properties:
    -----------
    sympy : sp.Integer
        The sympy representation of the integer.
    """

    def __init__(
        self,
        value: int = 1,
        parent: Optional[LambdaRingExpr] = None,
        id_: Hashable = None,
    ):
        self.value: int = value
        super().__init__(parent, id_)

    def __repr__(self) -> str:
        return str(self.value)

    def get_adams_var(self, i: int) -> int:
        """
        Returns the Adams variable of degree i.
        In this case, it is just its value.

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
        In this case it is the binomial coefficient
            binom(i+x-1,i)

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
        Applies an Adams operation of degree "degree" to any instances of this integer
        in the polynomial "ph". In this case, it leaves the polynomial unchanged.
        """
        return ph

    @dispatch(set)
    def _to_adams(self, operands: set[Operand]) -> sp.Expr:
        """
        Returns the integer.
        """
        return sp.Integer(self.value)

    def _subs_adams(self, ph: sp.Expr) -> sp.Expr:
        """
        Substitutes any instances of an Adams of this integer into its equivalent
        polynomial of lambdas. In this case, it leaves the polynomial unchanged.
        """
        return ph

    @property
    def sympy(self) -> sp.Integer:
        """
        The sympy representation of the integer.
        """
        return sp.Integer(self.value)
