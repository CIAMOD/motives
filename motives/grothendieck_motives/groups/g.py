import sympy as sp
from ..motive import Motive
from typing import Iterable
from ..lefschetz import Lefschetz
from ...core import LambdaRingContext
from multipledispatch import dispatch
from ...core.operand import Operand


class G(Motive, sp.AtomicExpr):
    def __new__(cls, alfas: Iterable[int], dim: int, *args, **kwargs):
        new_g = sp.AtomicExpr.__new__(cls)
        new_g._assumptions["commutative"] = True
        return new_g

    def __init__(self, alfas: Iterable[int], dim: int, *args, **kwargs):
        self.alfas = alfas
        self.dim = dim

        self.lef = Lefschetz()
        self._et_repr = self.lef**dim * sp.Mul(
            *[1 - self.lef ** (-alfa) for alfa in alfas],
        )
        self._lambda_vars = {}

    def get_adams_var(self, i: int) -> sp.Expr:
        return self.lef._to_adams(i, self._et_repr)

    def get_lambda_var(self, i: int) -> sp.Expr:
        if i not in self._lambda_vars:
            lrc = LambdaRingContext()

            ph_list = [self.get_adams_var(j) for j in range(i + 1)]

            self._lambda_vars[i] = lrc.get_adams_2_lambda_pol(i).xreplace(
                {lrc.adams_vars[i]: ph_list[i] for i in range(i + 1)}
            )

        return self._lambda_vars[i]

    @dispatch(int, sp.Expr)
    def _to_adams(self, degree: int, ph: sp.Expr) -> sp.Expr:
        """
        Applies the Adams operator to any instances of this group in the expression.

        This method raises an exception because groups should not appear directly
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
            Always raised as groups should be decomposed into their components.
        """
        raise Exception(
            f"There is a group in the expression {ph}. "
            "It should have been converted to its components."
        )

    @dispatch(set)
    def _to_adams(self, operands: set[Operand]) -> sp.Expr:
        """
        Converts this group into an equivalent Adams polynomial.

        Args:
        -----
        operands : set[Operand]
            The set of all operands in the expression tree.

        Returns:
        --------
        sp.Expr
            The Adams polynomial equivalent to this group.
        """
        return self._et_repr

    def _subs_adams(self, ph: sp.Expr) -> sp.Expr:
        """
        Substitutes Adams variables of this group in the expression
        with their equivalent Lambda polynomials.

        This method raises an exception because groups should not appear
        directly in the expression. They should be decomposed into their components.

        Args:
        -----
        ph : sp.Expr
            The polynomial in which to substitute the Adams variables.

        Raises:
        -------
        Exception
            Always raised as groups should be decomposed into their components.
        """
        raise Exception(
            f"There is a group in the expression {ph}. "
            "It should have been converted to its components."
        )

    @property
    def free_symbols(self):
        """
        Returns the set of free symbols in the group, which includes
        the Lefschetz motive.

        Returns:
        --------
        set[sp.Symbol]
            The set of free symbols in the group.
        """
        return {self.lef}
