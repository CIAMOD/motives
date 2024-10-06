from typing import TypeVar
from typeguard import typechecked
from multipledispatch import dispatch
import sympy as sp
from sympy.polys.rings import PolyRing
from sympy.polys.rings import PolyElement

from ...utils import expr_from_pol

from ...core.operator.nary_operator import Add
from ...core.lambda_context import LambdaContext
from ...core.operand import Operand

from ..motive import Motive
from ..lefschetz import Lefschetz

ET = TypeVar('ET')  # Define Operand as a TypeVar for type hinting

class Hodge(Motive):
    """
    An operand node in an expression tree that represents a Hodge motive.

    Parameters:
    -----------
    value : sympy.Symbol
        The value of the Hodge motive, i.e. how it is represented. It is
        the same as the value of its curve.
    g : int
        The genus of the Hodge motive. It is the same as the genus of its curve.
    parent : Node
        The parent node of the Hodge motive.
    id_ : hashable
        The id of the Hodge motive, used to identify it (if two operands
        have the same id, they are the same operand).

    Methods:
    --------
    sigma(degree: int) -> ET
        Applies the sigma operation to the current Hodge motive.
    lambda_(degree: int) -> ET
        Applies the lambda operation to the current Hodge motive.
    adams(degree: int) -> ET
        Applies the adams operation to the current Hodge motive.
    Z(t: Operand | int) -> ET
        Returns the generating function of the Hodge curve.

    Properties:
    -----------
    sympy : sp.Symbol
        The sympy representation of the Hodge motive.
    """

    def __init__(
        self, value: sp.Symbol | str, g: int = 1, parent=None, id_=None
    ) -> None:
        self.g: int = g
        if isinstance(value, str):
            value = sp.Symbol(value)
        self.value: sp.Symbol = value
        super().__init__(parent, id_)

        self.lambda_symbols: list[sp.Symbol] = [
            1,
            *[sp.Symbol(f"a{i}(h_{self.value})") for i in range(1, g + 1)],
        ]

        self._domain = sp.ZZ[[Lefschetz.L_VAR] + self.lambda_symbols[1:]]
        self._ring: PolyRing = self._domain.ring
        self._domain_symbols = [1] + [
            self._domain(var) for var in self.lambda_symbols[1:]
        ]
        self.lef_symbol = self._domain(Lefschetz.L_VAR)

        self._px_inv: list[sp.Expr] = [1]
        self._lambda_vars_pol: list[PolyElement] = [
            (
                self._domain_symbols[i]
                if i <= self.g
                else self._domain_symbols[2 * self.g - i]
                * self.lef_symbol ** (i - self.g)
            )
            for i in range(2 * self.g + 1)
        ]
        self._lambda_vars = [1] + [
            expr_from_pol(pol) for pol in self._lambda_vars_pol[1:]
        ]
        self._lambda_to_adams_pol: list[PolyElement] = [0]
        self.lambda_to_adams: list[sp.Expr] = [0]
        self._adams_vars: list[sp.Symbol] = [1, sp.Symbol(f"ψ_1(h_{self.value})")]

    def __repr__(self) -> str:
        return f"h_{self.value}"

    def _generate_adams_vars(self, n: int) -> None:
        """
        Generates the adams variables needed up to degree n.
        """
        self._adams_vars += [
            sp.Symbol(f"ψ_{i}(h_{self.value})")
            for i in range(len(self._adams_vars), n + 1)
        ]

    def _generate_lambda_vars(self, n: int) -> None:
        """
        Generates the lambda variables needed up to degree n.
        """
        self._lambda_vars_pol += [0] * (n - len(self._lambda_vars_pol) + 1)
        self._generate_inverse(n)

        for i in range(len(self._lambda_to_adams_pol), n + 1):
            self._lambda_to_adams_pol.append(
                self._ring.add(
                    *(
                        self._ring.mul(self._lambda_vars_pol[j], self._px_inv[i - j], j)
                        for j in range(1, i + 1)
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

    def get_lambda_var(self, i: int) -> sp.Symbol | int:
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
        return 0 if i >= len(self._lambda_vars) else self._lambda_vars[i]

    @typechecked
    def Z(self, t: Operand | int | ET) -> ET:
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
        l = Lefschetz()

        from ...core.expression_tree import ET as exp_tree  # Local import to avoid circular dependency
        et = exp_tree(
            Add(
                [
                    (
                        self.lambda_(i) * t**i
                        if i <= self.g
                        else self.lambda_(2 * self.g - i) * l ** (i - self.g) * t**i
                    )
                    for i in range(2 * self.g + 1)
                ]
            )
        )

        return et

    @dispatch(int, int)
    def _to_adams(self, degree: int, ph: int) -> int:
        """
        Catches the case where the polynomial is an integer.
        """
        return ph

    @dispatch(int, sp.Expr)
    def _to_adams(self, degree: int, ph: sp.Expr) -> sp.Expr:
        """
        Applies an adams operation of degree `degree` to any instances of this hodge
        motive in the polynomial `ph`.
        """
        max_adams = -1
        operands = ph.free_symbols
        for i, adams in reversed(list(enumerate(self._adams_vars))):
            if adams in operands:
                max_adams = i
                break

        return ph.xreplace(
            {
                self.get_adams_var(i): self.get_adams_var(degree * i)
                for i in range(1, max_adams + 1)
            }
        )

    @dispatch(set, LambdaContext)
    def _to_adams(
        self, operands: set[Operand], group_context: LambdaContext
    ) -> sp.Expr:
        """
        Returns the adams of degree 1 of the hodge motive.
        """
        return self.get_adams_var(1)

    def _generate_inverse(self, n: int) -> None:
        """
        Fills the inverse list with the first n elements of the inverse Hodge curve.
        """
        for m in range(len(self._px_inv), n + 1):
            self._px_inv.append(
                -self._ring.add(
                    self._ring.mul(self._lambda_vars_pol[i], self._px_inv[m - i])
                    for i in range(1, m + 1)
                )
            )

    def _subs_adams(self, group_context: LambdaContext, ph: sp.Expr) -> sp.Expr:
        """
        Substitutes any instances of an adams of this hodge motive (ψ_d(hodge)) into its
        equivalent polynomial of lambdas.
        """
        # Find the maximum adams degree of this variable in the polynomial
        max_adams = -1
        operands = ph.free_symbols
        for i, adams in reversed(list(enumerate(self._adams_vars))):
            if adams in operands:
                max_adams = i
                break

        # Generate the lambda_to_adams polynomials up to the degree needed
        self._generate_lambda_vars(len(self._adams_vars) - 1)

        for i in range(len(self.lambda_to_adams), max_adams + 1):
            self.lambda_to_adams.append(expr_from_pol(self._lambda_to_adams_pol[i]))

        # Substitute the adams variables into the polynomial
        ph = ph.xreplace(
            {
                self.get_adams_var(i): self.lambda_to_adams[i]
                for i in range(1, max_adams + 1)
            }
        )

        return ph

    @property
    def sympy(self) -> sp.Symbol:
        """
        The sympy representation of the curve motive.
        """
        return sp.Symbol(f"h_{self.value}")