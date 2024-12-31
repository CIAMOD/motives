import sympy as sp

from ..motive import Motive
from ..lefschetz import Lefschetz
from ...core.operand.operand import Operand

from .curvechow import CurveChow
from .curve import Curve


class Jacobian(Motive, sp.AtomicExpr):
    """
    Represents the Jacobian of a curve in an expression tree.

    The `Jacobian` is a motive associated with a `Curve` object. It supports Adams and Lambda operations,
    generating functions, and interacts with other motives like the CurveChow.

    Attributes:
    -----------
    curve : Curve
        The curve for which this Jacobian is defined.
    chow : CurveChow
        The Chow of the curve.
    g : int
        The genus of the curve.
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
        self.chow: CurveChow = curve.curve_chow
        self.g: int = curve.g

        self._adams_vars: dict[int, sp.Expr] = {}
        self._lambda_vars: dict[int, sp.Expr] = {}

        l = Lefschetz()
        self._et_repr: sp.Expr = sp.Add(
            *[
                (
                    self.chow.lambda_(i)
                    if i <= self.g
                    else self.chow.lambda_(2 * self.g - i) * l ** (i - self.g)
                )
                for i in range(2 * self.g + 1)
            ]
        )

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
        if i not in self._adams_vars:
            self._adams_vars[i] = self._et_repr.adams(i).to_adams()

        return self._adams_vars[i]

    def get_lambda_var(self, i: int) -> sp.Expr:
        """
        Returns the Jacobian with a Lambda operation applied to it.

        Args:
        -----
        i : int
            The degree of the Lambda operator.

        Returns:
        --------
        sp.Expr
            The Jacobian with the Lambda operator applied.
        """
        if i not in self._lambda_vars:
            self._lambda_vars[i] = self._et_repr.lambda_(i).to_lambda()

        return self._lambda_vars[i]

    @property
    def free_symbols(self) -> set[sp.Symbol]:
        """
        Returns the set of free symbols in the Jacobian.

        Returns:
        --------
        set[sp.Symbol]
            The set of free symbols in the Jacobian.
        """
        return {self.chow, Lefschetz()}

    def _apply_adams(self, degree: int, ph: sp.Expr) -> sp.Expr:
        """
        Applies the Adams operator to any instances of this Jacobian in the expression.

        This method raises an exception because Jacobians should not appear directly
        in the expression. Instead, they should be decomposed into their components.

        Args:
        -----
        degree : int
            The degree of the Adams operator to apply.
        ph : sp.Expr
            The polynomial in which to apply the Adams operator.

        Raises:
        -------
        Exception
            Always raised as Jacobians should be decomposed into their components.
        """
        raise Exception(
            f"There is a motive in the expression {ph}. "
            "It should have been converted to its components."
        )

    def _to_adams(self, operands: set[Operand]) -> sp.Expr:
        """
        Converts this Jacobian into an equivalent Adams polynomial.

        Args:
        -----
        operands : set[Operand]
            The set of all operands in the expression tree.

        Returns:
        --------
        sp.Expr
            The Adams polynomial equivalent to this Jacobian.
        """
        return self._et_repr.to_adams()

    def _subs_adams(self, ph: sp.Expr) -> sp.Expr:
        """
        Substitutes Adams variables in the polynomial with equivalent Lambda polynomials.

        Args:
        -----
        ph : sp.Expr
            The polynomial in which to substitute the Adams variables.

        Returns:
        --------
        sp.Expr
            The polynomial with Adams variables substituted by Lambda polynomials.

        Raises:
        -------
        Exception
            Always raised as Jacobians should be decomposed into their components.
        """
        raise Exception(
            f"There is a Jacobian in the expression {ph}. "
            "It should have been converted to its components."
        )
