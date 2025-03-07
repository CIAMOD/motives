from typeguard import typechecked
import sympy as sp
from sympy.polys.rings import PolyRing, PolyElement
from typing import Union

from ...utils import expr_from_pol, preorder_traversal

from ...core.operator.ring_operator import Adams, Lambda_

from ..motive import Motive
from ..lefschetz import Lefschetz


class CurveChow(Motive, sp.AtomicExpr):
    """
    Represents a Chow motive for a curve in an expression tree.

    A `CurveChow` object is initialized with a name and genus (`g`), and it allows the generation
    of Adams and Lambda variables based on the genus. It supports operations such as applying
    Adams or Lambda operators, generating Adams and Lambda variables, and computing generating functions.

    Attributes:
    -----------
    g : int
        The genus of the curve, which defines the degree limits of the Adams and Lambda operators.
    name : str
        The name of the CurveChow motive.
    lambda_symbols : list[sp.AtomicExpr]
        A list of Lambda symbols generated for the CurveChow motive.
    _domain : sp.Domain
        The domain used for the polynomial ring of Lambda operators.
    _ring : PolyRing
        The polynomial ring for the CurveChow motive.
    _domain_symbols : list[PolyElement]
        The list of domain symbols used in the polynomial ring.
    _lef_symbol : PolyElement
        The Lefschetz motive symbol.
    _px_inv : list[sp.Expr]
        The list of inverse expressions used for Adams and Lambda conversions.
    _lambda_vars_pol : list[PolyElement]
        The list of polynomials representing Lambda variables.
    _lambda_to_adams_pol : list[PolyElement]
        The list of polynomials for converting Lambda to Adams variables.
    _lambda_to_adams : list[sp.Expr]
        The list of expressions for converting Lambda to Adams variables.
    """

    is_commutative = True
    is_real = True

    def __new__(cls, name: str, g: int = 1, *args, **kwargs):
        """
        Creates a new instance of `CurveChow`.

        Args:
        -----
        name : str
            The name of the CurveChow motive.
        g : int, optional
            The genus of the curve, default is 1.

        Returns:
        --------
        CurveChow
            A new instance of `CurveChow`.
        """
        new_chow = sp.AtomicExpr.__new__(cls)
        new_chow._assumptions["commutative"] = True
        return new_chow

    def __init__(self, name: str, g: int = 1, *args, **kwargs) -> None:
        """
        Initializes a `CurveChow` instance.

        Args:
        -----
        name : str
            The name of the CurveChow motive.
        g : int, optional
            The genus of the curve, default is 1.
        """
        self.g: int = g
        self.name: str = name

        # Lambda symbols and domain setup
        self.lambda_symbols: list[sp.AtomicExpr] = [
            sp.Integer(1),
            self,
            *[sp.Symbol(f"a{i}({self})") for i in range(2, g + 1)],
        ]

        self._domain: sp.Domain = sp.ZZ[[Lefschetz()] + self.lambda_symbols[1:]]
        self._ring: PolyRing = self._domain.ring
        self._domain_symbols: list[PolyElement] = [self._domain.one] + [
            self._domain(var) for var in self.lambda_symbols[1:]
        ]
        self._lef_symbol: PolyElement = self._domain(Lefschetz())

        # Adams and Lambda variable setup
        self._px_inv: list[sp.Expr] = [sp.Integer(1)]
        self._lambda_vars_pol: list[PolyElement] = [
            (
                self._domain_symbols[i]
                if i <= self.g
                else self._domain_symbols[2 * self.g - i]
                * self._lef_symbol ** (i - self.g)
            )
            for i in range(2 * self.g + 1)
        ]
        self._lambda_vars: list[sp.Expr] = [
            self.lambda_symbols[i] for i in range(self.g + 1)
        ] + [
            self.lambda_symbols[2 * self.g - i] * Lefschetz() ** (i - self.g)
            for i in range(self.g + 1, 2 * self.g + 1)
        ]
        self._lambda_to_adams_pol: list[PolyElement] = [self._domain.zero]
        self._lambda_to_adams_function: list[sp.Expr] = [sp.Integer(0)]
        self._lambda_to_adams_symbol: list[sp.Expr] = [sp.Integer(0)]

    def __repr__(self) -> str:
        """
        Returns the string representation of the CurveChow motive.

        Returns:
        --------
        str
            A string representation in the form of "h{g}_{name}".
        """
        return f"h{self.g}_{self.name}"

    def _hashable_content(self) -> tuple:
        """
        Returns the hashable content of the CurveChow motive.

        Returns:
        --------
        tuple
            A tuple containing the name and genus.
        """
        return (self.name, self.g)

    def _generate_lambda_vars(self, n: int) -> None:
        """
        Generates the necessary Lambda variables up to degree `n`.

        Args:
        -----
        n : int
            The maximum degree of Lambda variables to generate.
        """
        self._lambda_vars_pol += [self._domain.zero] * (
            n - len(self._lambda_vars_pol) + 1
        )
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

    def get_adams_var(self, i: int, as_symbol: bool = False) -> sp.Expr:
        """
        Returns the CurveChow motive with an Adams operation applied to it.

        Args:
        -----
        i : int
            The degree of the Adams operator.
        as_symbol : bool, optional
            If True, returns the Adams variable as a SymPy Symbol. Otherwise, returns it as an
            Adams object.

        Returns:
        --------
        sp.Expr
            The CurveChow motive with the Adams operator applied.
        """
        if i == 0:
            return 1
        if i == 1:
            return self
        if as_symbol is False:
            return self.adams(i)
        return sp.Symbol(f"ψ{i}({self})")

    def get_lambda_var(self, i: int, as_symbol: bool = False) -> sp.Expr:
        """
        Returns the CurveChow motive with a Lambda operation applied to it.

        For Lambda operations with degree greater than 2g, the result is 0.

        Args:
        -----
        i : int
            The degree of the Lambda operator.
        as_symbol : bool, optional
            If True, returns the lambda variable as a SymPy Symbol. Otherwise, returns it as a
            Lambda_ object.

        Returns:
        --------
        sp.Expr
            The CurveChow motive with the Lambda operator applied, or 0 if the degree exceeds 2g.
        """
        return (
            sp.Integer(0)
            if i > 2 * self.g
            else (self.lambda_(i) if as_symbol is False else self._lambda_vars[i])
        )

    @typechecked
    def Z(self, t: int | sp.Expr) -> sp.Expr:
        """
        Computes the generating function of the CurveChow curve.

        The generating function is Σ_{i=0}^{2g} λ_i * t^i.

        Args:
        -----
        t : int or sp.Expr
            The variable to use in the generating function.

        Returns:
        --------
        sp.Expr
            The generating function of the CurveChow curve.
        """
        l = Lefschetz()

        et = sp.Add(
            *[
                (
                    self.lambda_(i) * t**i
                    if i <= self.g
                    else self.lambda_(2 * self.g - i) * l ** (i - self.g) * t**i
                )
                for i in range(2 * self.g + 1)
            ]
        )

        return et

    def _apply_adams(
        self, degree: int, ph: sp.Expr, max_adams_degree: int, as_symbol: bool = False
    ) -> sp.Expr:
        """
        Applies Adams operations to any instances of this CurveChow motive in a polynomial.

        Args:
        -----
        degree : int
            The degree of the Adams operator to apply.
        ph : sp.Expr
            The polynomial in which the Adams operator is applied.
        max_adams_degree : int
            The maximum degree of Adams operators in the expression.
        as_symbol : bool, optional
            If True, represents Adams operators as symbols.

        Returns:
        --------
        sp.Expr
            The polynomial with Adams operators applied to the CurveChow motive.
        """
        return ph.xreplace(
            {
                self.get_adams_var(i, as_symbol=as_symbol): self.get_adams_var(
                    degree * i, as_symbol=as_symbol
                )
                for i in range(1, max_adams_degree + 1)
            }
        )

    def _generate_inverse(self, n: int) -> None:
        """
        Fills the inverse list up to degree `n`.

        Args:
        -----
        n : int
            The maximum degree of the inverse needed.
        """
        for m in range(len(self._px_inv), n + 1):
            self._px_inv.append(
                -self._ring.add(
                    self._ring.mul(self._lambda_vars_pol[i], self._px_inv[m - i])
                    for i in range(1, m + 1)
                )
            )

    def _subs_adams(
        self, ph: sp.Expr, max_adams_degree: int, as_symbol: bool = False
    ) -> sp.Expr:
        """
        Substitutes Adams variables for equivalent Lambda polynomials in the given polynomial.

        This method is called during the `to_lambda` process to convert Adams variables
        that appear after converting the expression tree to an Adams polynomial.

        Args:
        -----
        ph : sp.Expr
            The polynomial in which to substitute the Adams variables.
        max_adams_degree : int
            The maximum degree of Adams operators in the expression.
        as_symbol : bool, optional
            If True, represents Lambda operators as symbols.

        Returns:
        --------
        sp.Expr
            The polynomial with Adams variables substituted by equivalent Lambda polynomials.
        """
        # Generate the lambda_to_adams polynomials up to the degree needed
        self._generate_lambda_vars(max_adams_degree)

        # Generate the lambda_to_adams equation with symbols
        for i in range(len(self._lambda_to_adams_symbol), max_adams_degree + 1):
            self._lambda_to_adams_symbol.append(
                expr_from_pol(self._lambda_to_adams_pol[i])
            )

        # If needed, sub the lambda symbols for the functions
        if as_symbol is False:
            for i in range(len(self._lambda_to_adams_function), max_adams_degree + 1):
                self._lambda_to_adams_function.append(
                    self._lambda_to_adams_symbol[i].xreplace(
                        {
                            self.get_lambda_var(j, as_symbol=True): self.get_lambda_var(
                                j, as_symbol=False
                            )
                            for j in range(len(self.lambda_symbols))
                        }
                    )
                )

        # Substitute the Adams variables into the polynomial
        ph = ph.xreplace(
            {
                self.get_adams_var(i, as_symbol): (
                    self._lambda_to_adams_function[i]
                    if as_symbol is False
                    else self._lambda_to_adams_symbol[i]
                )
                for i in range(1, max_adams_degree + 1)
            }
        )

        return ph
