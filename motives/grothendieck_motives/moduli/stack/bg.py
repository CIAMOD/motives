from ...motive import Motive
from ...groups.semisimple_g import SemisimpleG
from ...curves.curve import Curve
from ...lefschetz import Lefschetz
from ....core.operand.operand import Operand

import sympy as sp


class BunG(Motive, sp.AtomicExpr):
    """
    Represents the moduli stack of vector bundles on a curve in an expression tree.

    The `BunG` is a motive associated with a `Curve` object and a general group `G`.
    It supports Adams and Lambda operations, generating functions, and interacts with other motives.

    Attributes:
    -----------
    curve : Curve
        The curve for which this BunG is defined.
    group : G
        The general group associated with this BunG.
    lef : Lefschetz
        The Lefschetz motive.
    """

    def __new__(cls, curve: Curve, group: SemisimpleG, *args, **kwargs):
        """
        Creates a new instance of the `BunG` class.

        Args:
        -----
        curve : Curve
            The curve for which to create the BunG.
        group : G
            The general group.

        Returns:
        --------
        BunG
            A new instance of the `BunG` class.
        """
        new_bun = sp.AtomicExpr.__new__(cls)
        new_bun._assumptions["commutative"] = True
        return new_bun

    def __init__(self, curve: Curve, group: SemisimpleG, *args, **kwargs) -> None:
        """
        Initializes a `BunG` instance.

        Args:
        -----
        curve : Curve
            The curve for which this BunG is defined.
        group : G
            The general group.
        """
        self.curve: Curve = curve
        self.group: SemisimpleG = group
        self.lef: Lefschetz = Lefschetz()

        self._et_repr: sp.Expr = self.lef ** (
            (self.curve.g - 1) * sum(self.group.ds)
        ) * sp.Mul(*[self.curve.Z(self.lef ** (-d)) for d in self.group.ds])

        self._lambda_vars: dict[int, sp.Expr] = {}
        self._adams_vars: dict[int, sp.Expr] = {}

    def __repr__(self) -> str:
        """
        Returns the string representation of the BunG.

        Returns:
        --------
        str
            A string representation in the form of "BunG_{curve}_{group}".
        """
        return f"BunG_{self.curve}_{self.group}"

    def _hashable_content(self) -> tuple:
        """
        Returns the hashable content of the BunG.

        Returns:
        --------
        tuple
            A tuple containing the curve and the general linear group.
        """
        return (self.curve, self.group)

    @property
    def free_symbols(self) -> set:
        """
        Returns the set of free symbols in the BunG.

        Returns:
        --------
        set
            The set of free symbols in the BunG.
        """
        return self._et_repr.free_symbols

    def get_adams_var(self, i: int) -> sp.Expr:
        """
        Returns the BunG with an Adams operation applied to it.

        Args:
        -----
        i : int
            The degree of the Adams operator.

        Returns:
        --------
            The BunG with the Adams operator applied.
        """
        if i not in self._adams_vars:
            self._adams_vars[i] = self._et_repr.adams(i).to_adams()
        return self._adams_vars[i]

    def get_lambda_var(self, i: int) -> sp.Expr:
        """
        Returns the BunG with a Lambda operation applied to it.

        Args:
        -----
        i : int
            The degree of the Lambda operator.

        Returns:
        --------
        sp.Expr
            The BunG with the Lambda operator applied.
        """

        if i not in self._lambda_vars:
            self._lambda_vars[i] = self._et_repr.lambda_(i).to_lambda()
        return self._lambda_vars[i]

    def _apply_adams(self, degree: int, ph: sp.Expr) -> sp.Expr:
        """
        Applies the Adams operator to any instances of this BunG in the expression.

        Args:
        -----
        degree : int
            The degree of the Adams operator to apply.
        ph : sp.Expr
            The polynomial in which to apply the Adams operator.

        Returns:
        --------
        sp.Expr
            The polynomial with the Adams operator applied.

        Raises:
        -------
        Exception
            Always raised as BunGs should be decomposed into their components.
        """
        raise Exception(
            f"There is a BunG in the expression {ph}. "
            "It should have been converted to its components."
        )

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
            Always raised as BunGs should be decomposed into their components.
        """
        raise Exception(
            f"There is a BunG in the expression {ph}. "
            "It should have been converted to its components."
        )
