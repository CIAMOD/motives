from typeguard import typechecked
import sympy as sp
from sympy.polys.rings import PolyRing
from sympy.polys.rings import PolyElement

from ...utils import expr_from_pol

from ...core import LambdaRingContext
from ...core.operand import Operand

from ..motive import Motive
from ..lefschetz import Lefschetz
from ..point import Point

from .curvehodge import CurveHodge
from .curve import Curve

# TODO: Review this class and its methods
class Jacobian(Motive, sp.AtomicExpr):
    """
    Represents the Jacobian of a curve in an expression tree.

    The `Jacobian` is a motive associated with a `Curve` object. It supports Adams and Lambda operations,
    generating functions, and interacts with other motives like the CurveHodge.

    Attributes:
    -----------
    curve : Curve
        The curve for which this Jacobian is defined.
    g : int
        The genus of the curve.
    _ring : PolyRing
        The polynomial ring associated with the curve's CurveHodge motive.
    _lambda_vars : dict[int, sp.Expr]
        A dictionary storing Lambda variables for different degrees.
    _lambda_vars_pol : list[PolyElement]
        A list storing the polynomial representations of Lambda variables.
    _adams_vars : list[sp.AtomicExpr]
        A list storing the Adams variables for the Jacobian.
    """

    def __new__(cls, curve: Curve, *args, **kwargs):
        """
        Creates a new instance of the `Jacobian` class.

        Args:
        -----
        curve : Curve
            The curve for which to create the Jacobian.

        Returns:
        --------
        Jacobian
            A new instance of the `Jacobian` class.
        """
        new_jacobian = sp.AtomicExpr.__new__(cls)
        new_jacobian._assumptions["commutative"] = True
        return new_jacobian

    def __init__(self, curve: Curve, *args, **kwargs) -> None:
        """
        Initializes a `Jacobian` instance.

        Args:
        -----
        curve : Curve
            The curve for which this Jacobian is defined.
        """
        self.curve: Curve = curve
        self.g: int = curve.g

        self._ring: PolyRing = self.curve.curve_hodge._domain.ring

        self._lambda_vars: dict[int, sp.Expr] = {}
        self._lambda_vars_pol: list[PolyElement] = []
        self._adams_vars: list[sp.AtomicExpr] = [sp.Integer(1), self]

    def __repr__(self) -> str:
        """
        Returns the string representation of the Jacobian.

        Returns:
        --------
        str
            A string representation in the form of "Jacobian_{curve}".
        """
        return f"Jacobian_{self.curve}"

    def _hashable_content(self) -> tuple:
        """
        Returns the hashable content of the Jacobian.

        Returns:
        --------
        tuple
            A tuple containing the curve.
        """
        return (self.curve,)

    def _generate_adams_vars(self, n: int) -> None:
        """
        Generates the Adams variables needed for the Jacobian up to degree `n`.

        Args:
        -----
        n : int
            The maximum degree of Adams needed.
        """
        self.curve.curve_hodge._generate_adams_vars(n)
        self._adams_vars += [
            sp.Symbol(f"ψ_{i}({self})") for i in range(len(self._adams_vars), n + 1)
        ]

    def _generate_lambda_vars(self, n: int) -> None:
        """
        Generates the Lambda variables needed for the Jacobian up to degree `n`.

        Args:
        -----
        n : int
            The maximum degree of Lambda needed.
        """
        self.curve.curve_hodge._generate_lambda_vars(n)

        if len(self._lambda_vars_pol) == 0:
            self._lambda_vars_pol = [self._ring.one]

        # The Jacobian is the sum over all degrees of the symmetric powers of the CurveHodge
        for i in range(1, n + 1):
            self._lambda_vars_pol.append(
                self._lambda_vars_pol[i - 1] + self.curve.curve_hodge._lambda_vars_pol[i]
            )

    def get_adams_var(self, i: int) -> sp.Expr:
        """
        Returns the Jacobian with an Adams operation applied to it.

        Args:
        -----
        i : int
            The degree of the Adams operator.

        Returns:
        --------
        sp.Expr
            The Jacobian with the Adams operator applied.
        """
        self._generate_adams_vars(i)
        return self._adams_vars[i]

    def get_lambda_var(self, i: int, context: LambdaRingContext = None) -> sp.Expr:
        """
        Returns the Jacobian with a Lambda operation applied to it.

        Args:
        -----
        i : int
            The degree of the Lambda operator.
        context : LambdaRingContext, optional
            The ring context used for the conversion between operators.

        Returns:
        --------
        sp.Expr
            The Jacobian with the Lambda operator applied.
        """
        self._generate_lambda_vars(i)

        if i not in self._lambda_vars:
            self._lambda_vars[i] = expr_from_pol(self._lambda_vars_pol[i])

        return self._lambda_vars[i]

    @typechecked
    def Z(self, t: int | sp.Expr) -> sp.Expr:
        """
        Computes the generating function of the Jacobian.

        Args:
        -----
        t : int or sp.Expr
            The variable to use in the generating function.

        Returns:
        --------
        sp.Expr
            The generating function of the Jacobian.
        """
        return self.curve.curve_hodge.Z(1) / (1 - t)

    @property
    def free_symbols(self) -> set[sp.Symbol]:
        """
        Returns the set of free symbols in the Jacobian.

        Returns:
        --------
        set[sp.Symbol]
            The set of free symbols in the Jacobian.
        """
        return {self}

    def _subs_adams(self, lrc: LambdaRingContext, ph: sp.Expr) -> sp.Expr:
        """
        Substitutes Adams variables in the polynomial with equivalent Lambda polynomials.

        Args:
        -----
        lrc : LambdaRingContext
            The Grothendieck ring context used for the conversion between operators.
        ph : sp.Expr
            The polynomial in which to substitute the Adams variables.

        Returns:
        --------
        sp.Expr
            The polynomial with Adams variables substituted by Lambda polynomials.
        """
        ph = self.curve.curve_hodge._subs_adams(lrc, ph)
        return ph
