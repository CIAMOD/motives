from typeguard import typechecked
from typing import TypeVar

Jacobian = TypeVar("Jacobian")

import sympy as sp
from sympy.polys.rings import PolyRing
from sympy.polys.rings import PolyElement

from ...utils import expr_from_pol

from ...core.operand.operand import Operand

from ..motive import Motive
from ..lefschetz import Lefschetz
from ..point import Point

from .curvechow import CurveChow


class Curve(Motive, sp.AtomicExpr):
    """
    Represents the motivic class of an abstract smooth complex algebraic curve
    of genus g in the an expression tree in the Grothendieck ring of varieties or Chow motives
    (or any extension of completion of them).

    The motive of a `Curve` X is represented through its Chow decomposition, as the  sum of a point,
    the Lefschetz motive and, h^1(X), the `CurveChow` componente of the curve.
    It supports Adams and lambda operations, generating functions,
    and Jacobian calculations.

    Attributes:
    -----------
    name : str
        The name of the curve.
    g : int
        The genus of the curve, which influences the CurveChow motive.
    point : Point
        The point motive of the curve.
    curve_chow : CurveChow
        The CurveChow motive of the curve.
    lefschetz : Lefschetz
        The Lefschetz motive of the curve.
    _ring : PolyRing
        The polynomial ring of the curve's CurveChow motive.
    _lambda_vars : dict[int, sp.Expr]
        A dictionary storing Lambda variables for different degrees.
    _lambda_vars_pol : list[PolyElement]
        A list storing the polynomial representations of Lambda variables.
    _adams_vars : list[sp.AtomicExpr]
        A list storing the Adams variables for the curve.
    """

    def __new__(cls, name: sp.Symbol, g: int = 1, *args, **kwargs):
        """
        Creates a new instance of the `Curve` class.

        Args:
        -----
        name : sp.Symbol
            The name of the curve.
        g : int, optional
            The genus of the curve, default is 1.

        Returns:
        --------
        Curve
            A new instance of the `Curve` class.
        """
        new_curve = sp.AtomicExpr.__new__(cls)
        new_curve._assumptions["commutative"] = True
        return new_curve

    def __init__(self, name: str, g: int = 1, *args, **kwargs) -> None:
        """
        Initializes a `Curve` instance.

        Args:
        -----
        name : str
            The name of the curve.
        g : int, optional
            The genus of the curve, default is 1.
        """
        self.g: int = g
        self.name: str = name

        self.point: Point = Point()
        self.curve_chow: CurveChow = CurveChow(name, g)
        self.lefschetz: Lefschetz = Lefschetz()

        self._ring: PolyRing = self.curve_chow._domain.ring

        self._lambda_vars: dict[int, sp.Expr] = {}
        self._lambda_vars_pol: list[PolyElement] = []
        self._adams_vars: list[sp.AtomicExpr] = [sp.Integer(1), self]

    def __repr__(self) -> str:
        """
        Returns the string representation of the curve.

        Returns:
        --------
        str
            A string representation in the form of "C{g}_{name}".
        """
        return f"C{self.g}_{self.name}"

    def _hashable_content(self) -> tuple:
        """
        Returns the hashable content of the curve.

        Returns:
        --------
        tuple
            A tuple containing the name and genus.
        """
        return (self.name, self.g)

    def _generate_adams_vars(self, n: int) -> None:
        """
        Generates the Adams variables needed for the curve up to degree `n`.

        Args:
        -----
        n : int
            The maximum degree of Adams needed.
        """
        self.curve_chow._generate_adams_vars(n)
        self._adams_vars += [
            sp.Symbol(f"ψ_{i}({self})") for i in range(len(self._adams_vars), n + 1)
        ]

    def _generate_lambda_vars(self, n: int) -> None:
        """
        Generates the Lambda variables needed for the curve up to degree `n`.

        Computes the Lambda variables by applying the convolution formula for the Lambda of a sum.

        Args:
        -----
        n : int
            The maximum degree of Lambda needed.
        """
        self.curve_chow._generate_lambda_vars(n)

        if len(self._lambda_vars_pol) == 0:
            self._lambda_vars_pol = [self._ring.one]

        for i in range(len(self._lambda_vars_pol), n + 1):
            self._lambda_vars_pol.append(
                self._lambda_vars_pol[i - 1]
                + self._ring.add(
                    *(
                        self._ring.mul(
                            self.curve_chow._lambda_vars_pol[i - j],
                            self.curve_chow._lef_symbol**j,
                        )
                        for j in range(i + 1)
                    )
                )
            )

    def get_adams_var(self, i: int) -> sp.Expr:
        """
        Returns the curve with an Adams operation applied to it.

        Args:
        -----
        i : int
            The degree of the Adams operator.

        Returns:
        --------
        sp.Expr
            The curve with the Adams operator applied.
        """
        self._generate_adams_vars(i)
        return self._adams_vars[i]

    def get_lambda_var(self, i: int) -> sp.Expr:
        """
        Returns the curve with a Lambda operation applied to it.

        Args:
        -----
        i : int
            The degree of the Lambda operator.

        Returns:
        --------
        sp.Expr
            The curve with the Lambda operator applied.
        """
        self._generate_lambda_vars(i)

        if i not in self._lambda_vars:
            self._lambda_vars[i] = expr_from_pol(self._lambda_vars_pol[i])

        return self._lambda_vars[i]

    @typechecked
    def P(self, t: int | sp.Expr) -> sp.Expr:
        """
        Computes the generating function of the CurveChow motive of the curve.

        Args:
        -----
        t : int or sp.Expr
            The variable to use in the generating function.

        Returns:
        --------
        sp.Expr
            The generating function of the CurveChow motive.
        """
        return self.curve_chow.Z(t)

    @property
    def Jac(self) -> Jacobian:
        """
        Returns the Jacobian object associated with the curve.

        Returns:
        --------
        Jacobian
            The Jacobian of the curve.
        """
        from .jacobian import Jacobian

        return Jacobian(self)

    @typechecked
    def Z(self, t: int | sp.Expr) -> sp.Expr:
        """
        Computes the generating function of the curve.

        Args:
        -----
        t : int or sp.Expr
            The variable to use in the generating function.

        Returns:
        --------
        sp.Expr
            The generating function of the curve.
        """
        return self.P(t) / ((1 - t) * (1 - self.lefschetz * t))

    def _apply_adams(self, degree: int, ph: sp.Expr) -> sp.Expr:
        """
        Applies the Adams operator to any instances of this curve in the polynomial.

        Raises an exception, as curves should be decomposed into their components.

        Args:
        -----
        degree : int
            The degree of the Adams operator to apply.
        ph : sp.Expr
            The polynomial in which to apply the Adams operator.

        Raises:
        -------
        Exception
            Always raised as curves should not be directly used in the expression.
        """
        raise Exception(
            f"There is a curve in the expression {ph}. "
            "It should have been converted to its components."
        )

    def _to_adams(self, operands: set[Operand]) -> sp.Expr:
        """
        Converts this curve into an equivalent Adams polynomial.

        Args:
        -----
        operands : set[Operand]
            The set of all operands in the expression tree.

        Returns:
        --------
        sp.Expr
            A polynomial of Adams operators equivalent to this curve.
        """
        return (
            self.curve_chow._to_adams(operands)
            + self.lefschetz._to_adams(operands)
            + self.point._to_adams(operands)
        )

    def _subs_adams(self, ph: sp.Expr) -> sp.Expr:
        """
        Substitutes Adams variables in the polynomial with equivalent Lambda polynomials.

        This method is called during the `to_lambda` process to replace Adams variables
        that appear after converting the expression tree to an Adams polynomial.

        Args:
        -----
        ph : sp.Expr
            The polynomial in which to substitute the Adams variables.

        Returns:
        --------
        sp.Expr
            The polynomial with Adams variables substituted by Lambda polynomials.
        """
        ph = self.curve_chow._subs_adams(ph)
        return ph

    @property
    def free_symbols(self) -> set[sp.Symbol]:
        """
        Returns the set of free symbols in the curve, which includes the CurveChow, Lefschetz, and Point motives.

        Returns:
        --------
        set[sp.Symbol]
            The set of free symbols in the curve.
        """
        return {self.curve_chow, self.lefschetz, self.point}
