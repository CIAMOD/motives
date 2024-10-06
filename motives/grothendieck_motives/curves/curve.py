from typing import Optional, Hashable, TypeVar
from typeguard import typechecked
from multipledispatch import dispatch
import sympy as sp
from sympy.polys.rings import PolyRing
from sympy.polys.rings import PolyElement

ET = TypeVar('ET')  # Define Operand as a TypeVar for type hinting

from ...utils import expr_from_pol

from ...core.lambda_context import LambdaContext
from ...core.node import Node
from ...core.operand import Operand

from ..motive import Motive
from ..lefschetz import Lefschetz
from ..point import Point

from .hodge import Hodge


class Curve(Motive):
    """
    An operand in an expression tree that represents a curve.

    Parameters:
    -----------
    value : sympy.Symbol
        The value of the curve, i.e. how it is represented.
    g : int
        The genus of the curve.
    parent : Node
        The parent node of the curve.
    id_ : hashable
        The id of the curve, used to identify it (if two operands
        have the same id, they are the same operand).

    Methods:
    --------
    sigma(degree: int) -> ET
        Applies the sigma operation to the current curve.
    lambda_(degree: int) -> ET
        Applies the lambda operation to the current curve.
    adams(degree: int) -> ET
        Applies the adams operation to the current curve.
    P(t: Variable) -> ET
        Returns the generating function of the hodge curve.
    Z(t: Variable) -> ET
        Returns the generating function of the curve.

    Properties:
    -----------
    sympy : sp.Symbol
        The sympy representation of the curve.
    Jac : ET
        The Jacobian of the curve.
    """

    def __init__(
        self,
        value: sp.Symbol | str,
        g: int = 1,
        parent: Optional[Node] = None,
        id_: Hashable = None,
    ) -> None:
        self.g: int = g
        if isinstance(value, str):
            value = sp.Symbol(value)
        self.value: sp.Symbol = value
        super().__init__(parent, id_)

        self.point: Point = Point()
        self.curve_hodge: Hodge = Hodge(value, g, id_=self.id + "_hodge")
        self.lefschetz: Lefschetz = Lefschetz()

        self._ring: PolyRing = self.curve_hodge._domain.ring

        self.jac: sp.Expr | None = None
        self._lambda_vars: dict[int, sp.Expr] = {}
        self._lambda_vars_pol: list[PolyElement] = []
        self._adams_vars: list[sp.Symbol] = [1]

    def __repr__(self) -> str:
        return str(self.value)

    def _generate_adams_vars(self, n: int) -> None:
        """
        Generates the adams variables needed up to degree n.
        """
        self.curve_hodge._generate_adams_vars(n)
        self._adams_vars += [
            sp.Symbol(f"ψ_{i}(C_{self.value})")
            for i in range(len(self._adams_vars), n + 1)
        ]

    def _generate_lambda_vars(self, n: int) -> None:
        """
        Generates the lambda variables needed up to degree n.
        """
        self.curve_hodge._generate_lambda_vars(n)

        if len(self._lambda_vars_pol) == 0:
            self._lambda_vars_pol = [1]

        for i in range(len(self._lambda_vars_pol), n + 1):
            self._lambda_vars_pol.append(
                self._lambda_vars_pol[i - 1]
                + self._ring.add(
                    *(
                        self._ring.mul(
                            self.curve_hodge._lambda_vars_pol[i - j],
                            self.curve_hodge.lef_symbol**j,
                        )
                        for j in range(i + 1)
                    )
                )
            )

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
        self._generate_lambda_vars(i)

        if i not in self._lambda_vars:
            self._lambda_vars[i] = expr_from_pol(self._lambda_vars_pol[i])

        return self._lambda_vars[i]

    @typechecked
    def P(self, t: Operand | int | ET) -> ET:
        """
        Returns the generating function of the hodge curve.

        Parameters
        ----------
        t : Operand or int or ET
            The variable to use in the generating function.

        Returns
        -------
        The generating function of the hodge curve.
        """
        return self.curve_hodge.Z(t)

    @property
    def Jac(self) -> ET:
        """
        The Jacobian of the curve.
        """
        if self.jac is None:
            self.jac = self.curve_hodge.Z(1)
        return self.jac

    @typechecked
    def Z(self, t: Operand | int | ET) -> ET:
        """
        Returns the generating function of the curve.

        Parameters
        ----------
        t : Operand or int or ET
            The variable to use in the generating function.

        Returns
        -------
        The generating function of the curve.
        """
        return self.P(t) / ((1 - t) * (1 - self.lefschetz * t))

    @dispatch(int, int)
    def _to_adams(self, degree: int, ph: int) -> int:
        """
        Catches the case where the polynomial is an integer.
        """
        return ph

    @dispatch(int, sp.Expr)
    def _to_adams(self, degree: int, ph: sp.Expr) -> sp.Expr:
        """
        Applies an adams operation of degree `degree` to any instances of this curve
        in the polynomial `ph`.
        """
        ph = self.curve_hodge._to_adams(degree, ph)
        return ph

    @dispatch(set, LambdaContext)
    def _to_adams(
        self, operands: set[Operand], group_context: LambdaContext
    ) -> sp.Expr:
        """
        Returns the adams of degree 1 of this curve.
        """
        return (
            self.curve_hodge._to_adams(operands, group_context)
            + self.lefschetz._to_adams(operands, group_context)
            + self.point._to_adams(operands, group_context)
        )

    def _subs_adams(self, group_context: LambdaContext, ph: sp.Expr) -> sp.Expr:
        """
        Substitutes any instances of an adams of this curve (ψ_d(curve)) into its
        equivalent polynomial of lambdas.
        """
        ph = self.curve_hodge._subs_adams(group_context, ph)
        return ph

    @property
    def sympy(self) -> sp.Symbol:
        """
        The sympy representation of the curve.
        """
        return sp.Symbol(f"C_{self.value}_{self.g}")