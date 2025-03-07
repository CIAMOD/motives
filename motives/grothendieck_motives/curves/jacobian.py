import sympy as sp
from typing import Dict

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

    Attributes
    ----------
    curve : Curve
        The curve for which this Jacobian is defined.
    chow : CurveChow
        The Chow motive of the curve.
    g : int
        The genus of the curve.
    """

    is_commutative = True
    is_real = True

    def __new__(cls, curve: Curve, *args, **kwargs):
        """
        Creates a new instance of the `Jacobian` class.

        Parameters
        ----------
        curve : Curve
            The curve for which to create the Jacobian.

        Returns
        -------
        Jacobian
            A new instance of the `Jacobian` class.
        """
        new_jacobian = sp.AtomicExpr.__new__(cls)
        return new_jacobian

    def __init__(self, curve: Curve, *args, **kwargs) -> None:
        """
        Initializes a `Jacobian` instance.

        Parameters
        ----------
        curve : Curve
            The curve for which this Jacobian is defined.
        """
        self.curve: Curve = curve
        self.chow: CurveChow = curve.curve_chow
        self.g: int = curve.g

        self._adams_vars: Dict[int, sp.Expr] = {}
        self._lambda_vars: Dict[int, sp.Expr] = {}

        l = Lefschetz()
        self._et_repr: sp.Expr = sp.Add(
            *[
                (
                    self.chow.get_lambda_var(i)
                    if i <= self.g
                    else self.chow.get_lambda_var(2 * self.g - i) * l ** (i - self.g)
                )
                for i in range(2 * self.g + 1)
            ]
        )

    def __repr__(self) -> str:
        """
        Returns the string representation of the Jacobian.

        Returns
        -------
        str
            A string representation in the form of "Jacobian_{curve}".
        """
        return f"Jacobian_{self.curve}"

    def _hashable_content(self) -> tuple:
        """
        Returns the hashable content of the Jacobian.

        Returns
        -------
        tuple
            A tuple containing the curve.
        """
        return (self.curve,)

    def get_max_adams_degree(self) -> int:
        """
        Returns the maximum degree of the Adams operator for this Jacobian.

        Returns
        -------
        int
            The maximum degree of the Adams operator.
        """
        return self.g

    def get_adams_var(self, i: int, as_symbol: bool = False) -> sp.Expr:
        """
        Returns the Jacobian with an Adams operation applied to it.

        Parameters:
        -----
        i : int
            The degree of the Adams operator.
        as_symbol : bool, optional
            If True, returns the Adams variable as a SymPy Symbol. Otherwise, returns it as an
            Adams_ object.

        Returns:
        --------
        sp.Expr
            The Jacobian with the Adams operator applied.
        """
        if i not in self._adams_vars:
            self._adams_vars[i] = self._et_repr.adams(i).to_adams(as_symbol=True)

        if as_symbol is False:
            return self._adams_vars[i].xreplace(
                {
                    self.chow.get_adams_var(j, as_symbol=True): self.chow.get_adams_var(
                        j, as_symbol=False
                    )
                    for j in range(2, i * self.g + 1)
                }
            )

        return self._adams_vars[i]

    def get_lambda_var(self, i: int, as_symbol: bool = False) -> sp.Expr:
        """
        Returns the Jacobian with a Lambda operation applied to it.

        Parameters:
        -----
        i : int
            The degree of the Lambda operator.
        as_symbol : bool, optional
            If True, returns the Lambda variable as a SymPy Symbol. Otherwise, returns it as a
            Lambda_ object.

        Returns:
        --------
        sp.Expr
            The Jacobian with the Lambda operator applied.
        """
        if i not in self._lambda_vars:
            self._lambda_vars[i] = self._et_repr.lambda_(i).to_lambda(as_symbol=True)

        if as_symbol is False:
            return self._lambda_vars[i].xreplace(
                {
                    symbol: self.chow.get_lambda_var(j, as_symbol=False)
                    for j, symbol in enumerate(self.chow.lambda_symbols[2:], start=2)
                }
            )

        return self._lambda_vars[i]

    @property
    def free_symbols(self) -> set[sp.Symbol]:
        """
        Returns the set of free symbols in the Jacobian.

        Returns
        -------
        set of sp.Symbol
            The set of free symbols in the Jacobian.
        """
        return {self.chow, Lefschetz()}

    def _apply_adams(
        self, degree: int, ph: sp.Expr, max_adams_degree: int, as_symbol: bool = False
    ) -> sp.Expr:
        """
        Applies the Adams operator to any instances of this Jacobian in the expression.

        Parameters
        ----------
        degree : int
            The degree of the Adams operator to apply.
        ph : sp.Expr
            The polynomial in which to apply the Adams operator.
        max_adams_degree : int
            The maximum degree of Adams operators in the expression.
        as_symbol : bool, optional
            If True, represents Adams operators as symbols.

        Raises
        ------
        Exception
            Always raised as Jacobians should be decomposed into their components.
        """
        raise Exception(
            f"There is a motive in the expression {ph}. "
            "It should have been converted to its components."
        )

    def _subs_adams(
        self, ph: sp.Expr, max_adams_degree: int, as_symbol: bool = False
    ) -> sp.Expr:
        """
        Substitutes Adams variables in the polynomial with equivalent Lambda polynomials.

        Parameters
        ----------
        ph : sp.Expr
            The polynomial in which to substitute the Adams variables.
        max_adams_degree : int
            The maximum degree of Adams operators in the expression.
        as_symbol : bool, optional
            If True, represents Lambda operators as symbols.

        Raises
        ------
        Exception
            Always raised as Jacobians should be decomposed into their components.
        """
        raise Exception(
            f"There is a Jacobian in the expression {ph}. "
            "It should have been converted to its components."
        )
