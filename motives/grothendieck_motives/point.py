from typing import Hashable, Optional
from multipledispatch import dispatch
import sympy as sp

from ..core.lambda_context import LambdaContext
from ..core.operand import Operand
from ..core.node import Node

from .motive import Motive

class Point(Motive):
    """
    An operand node in an expression tree that represents a point motive.

    Parameters:
    -----------
    parent : Node
        The parent node of the point motive.
    id_ : hashable
        Because the point motive is unique, the id is always "point". It should
        not be changed.

    Methods:
    --------
    sigma(degree: int) -> ET
        Applies the sigma operation to the current point motive.
    lambda_(degree: int) -> ET
        Applies the lambda operation to the current point motive.
    adams(degree: int) -> ET
        Applies the adams operation to the current point motive.

    Properties:
    sympy : sp.Symbol
        The sympy representation of the point.
    """

    def __init__(self, parent: Optional[Node] = None, id_: Hashable = "point"):
        id_ = "point"
        super().__init__(parent, id_)

    def __repr__(self) -> str:
        return "pt"

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
        return 1

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
        return 1

    @dispatch(int, object)
    def _to_adams(self, degree: int, ph: sp.Expr) -> sp.Expr:
        """
        Applies an adams operation of degree `degree` to any instances of this point
        in the polynomial `ph`.
        """
        return ph

    @dispatch(set, LambdaContext)
    def _to_adams(
        self, operands: set[Operand], group_context: LambdaContext
    ) -> sp.Expr:
        """
        Returns the adams of degree 1 of the point.
        """
        return sp.Integer(1)

    def _subs_adams(self, group_context: LambdaContext, ph: sp.Expr) -> sp.Expr:
        """
        Substitutes any instances of an adams of this point into its equivalent
        polynomial of lambdas.
        """
        return ph

    @property
    def sympy(self) -> sp.Symbol:
        """
        The sympy representation of the point motive.
        """
        return sp.Symbol("pt")