from typing import Hashable, Optional
from multipledispatch import dispatch
import sympy as sp

from ..core.lambda_context import LambdaContext
from ..core.operand import Operand
from ..core.node import Node

from .motive import Motive

class Symbol(Motive):
    """
    An operand in an expression tree that represents an abstract one
    dimensional motive, like the Lefschetz motive.

    Parameters:
    -----------
    value : sympy.Symbol
        The value of the symbolic integer, i.e. how it is represented.
    parent : Node
        The parent node of the symbolic integer.
    id_ : hashable
        The id of the symbolic integer, used to identify it (if two operands
        have the same id, they are the same operand).

    Methods:
    --------
    sigma(degree: int) -> ET
        Applies the sigma operation to the current symbolic integer.
    lambda_(degree: int) -> ET
        Applies the lambda operation to the current symbolic integer.
    adams(degree: int) -> ET
        Applies the adams operation to the current symbolic integer.

    Properties:
    -----------
    sympy : sp.Symbol
        The sympy representation of the symbol.
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

    def __repr__(self) -> str:
        return str(self.value)

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
        return self.value

    def get_lambda_var(self, i: int) -> sp.Expr:
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
        return self.value**i

    @dispatch(int, int)
    def _to_adams(self, degree: int, ph: int) -> int:
        """
        Catches the case where the polynomial is an integer.
        """
        return ph

    @dispatch(int, sp.Expr)
    def _to_adams(self, degree: int, ph: sp.Expr) -> sp.Expr:
        """
        Applies an adams operation of degree `degree` to any instances of this symbol
        in the polynomial `ph`.
        """
        return ph.xreplace({self.value: self.value**degree})

    @dispatch(set, LambdaContext)
    def _to_adams(
        self, operands: set[Operand], group_context: LambdaContext
    ) -> sp.Expr:
        """
        Returns the adams of degree 1 of the symbol.
        """
        return self.value

    def _subs_adams(self, group_context: LambdaContext, ph: sp.Expr) -> sp.Expr:
        """
        Substitutes any instances of an adams of this symbol into its equivalent
        polynomial of lambdas.
        """
        return ph

    @property
    def sympy(self) -> sp.Symbol:
        """
        The sympy representation of the symbol.
        """
        return sp.Symbol(f"s_{self.value}")