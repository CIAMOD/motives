from typing import TypeVar
from typeguard import typechecked
from multipledispatch import dispatch
import sympy as sp
from sympy.polys.rings import PolyRing
from sympy.polys.rings import PolyElement

from ...utils import expr_from_pol

from ...core import GrothendieckRingContext
from ...core.operand import Operand

from ..motive import Motive
from ..lefschetz import Lefschetz

ET = TypeVar('ET')  # Define Operand as a TypeVar for type hinting

class Hodge(Motive, sp.AtomicExpr):
    """
    Represents a Hodge motive for a curve in an expression tree.

    A `Hodge` object is initialized with a name and genus (g), and it allows the generation
    of Adams and Lambda variables based on the genus. It supports operations such as applying
    Adams or Lambda operators, generating Adams and Lambda variables, and computing generating functions.

    Attributes:
    -----------
    g : int
        The genus of the curve, which defines the degree limits of the Adams and Lambda operators.
    name : str
        The name of the Hodge motive.
    lambda_symbols : list[sp.AtomicExpr]
        A list of Lambda symbols generated for the Hodge motive.
    _domain : sp.Domain
        The domain used for the polynomial ring of Lambda operators.
    _ring : PolyRing
        The polynomial ring for the Hodge motive.
    _domain_symbols : list[PolyElement]
        The list of domain symbols used in the polynomial ring.
    _lef_symbol : PolyElement
        The Lefschetz motive symbol.
    _px_inv : list[sp.Expr]
        The list of inverse expressions used for Adams and Lambda conversions.
    _lambda_vars_pol : list[PolyElement]
        The list of polynomials representing Lambda variables.
    _lambda_vars : list[sp.Expr]
        The list of expressions representing Lambda variables.
    _lambda_to_adams_pol : list[PolyElement]
        The list of polynomials for converting Lambda to Adams variables.
    _lambda_to_adams : list[sp.Expr]
        The list of expressions for converting Lambda to Adams variables.
    _adams_vars : list[sp.Expr]
        The list of Adams variables generated for the Hodge motive.
    """

    def __new__(cls, name: str, g: int = 1, *args, **kwargs):
        """
        Creates a new instance of `Hodge`.

        Args:
        -----
        name : str
            The name of the Hodge motive.
        g : int, optional
            The genus of the curve, default is 1.

        Returns:
        --------
        Hodge
            A new instance of `Hodge`.
        """
        new_hodge = sp.AtomicExpr.__new__(cls)
        new_hodge._assumptions["commutative"] = True
        return new_hodge

    def __init__(self, name: str, g: int = 1, *args, **kwargs):
        """
        Initializes a `Hodge` instance.

        Args:
        -----
        name : str
            The name of the Hodge motive.
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
        self._lambda_vars = [sp.Integer(1)] + [
            expr_from_pol(pol) for pol in self._lambda_vars_pol[1:]
        ]
        self._lambda_to_adams_pol: list[PolyElement] = [self._domain.zero]
        self._lambda_to_adams: list[sp.Expr] = [sp.Integer(0)]
        self._adams_vars: list[sp.Expr] = [sp.Integer(1), self]

    def __repr__(self) -> str:
        """
        Returns the string representation of the Hodge motive.

        Returns:
        --------
        str
            A string representation in the form of "h{g}_{name}".
        """
        return f"h{self.g}_{self.name}"

    def _hashable_content(self) -> tuple:
        """
        Returns the hashable content of the Hodge motive.

        Returns:
        --------
        tuple
            A tuple containing the name and genus.
        """
        return (self.name, self.g)

    def _generate_adams_vars(self, n: int) -> None:
        """
        Generates the necessary Adams variables up to degree `n`.

        Args:
        -----
        n : int
            The maximum degree of Adams variables to generate.
        """
        self._adams_vars += [
            sp.Symbol(f"ψ_{i}({self})") for i in range(len(self._adams_vars), n + 1)
        ]

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

    def get_adams_var(self, i: int) -> sp.Expr:
        """
        Returns the Hodge motive with an Adams operation applied to it.

        Args:
        -----
        i : int
            The degree of the Adams operator.

        Returns:
        --------
        sp.Expr
            The Hodge motive with the Adams operator applied.
        """
        self._generate_adams_vars(i)
        return self._adams_vars[i]

    def get_lambda_var(self, i: int, context: GrothendieckRingContext = None) -> sp.Expr:
        """
        Returns the Hodge motive with a Lambda operation applied to it.

        For Lambda operations with degree greater than 2g, the result is 0.

        Args:
        -----
        i : int
            The degree of the Lambda operator.
        context : GrothendieckRingContext, optional
            The ring context used for conversion between operators.

        Returns:
        --------
        sp.Expr
            The Hodge motive with the Lambda operator applied, or 0 if the degree exceeds 2g.
        """
        return sp.Integer(0) if i >= len(self._lambda_vars) else self._lambda_vars[i]

    @typechecked
    def Z(self, t: int | sp.Expr) -> sp.Expr:
        """
        Computes the generating function of the Hodge curve.

        The generating function is Σ_{i=0}^{2g} λ_i * t^i.

        Args:
        -----
        t : int or sp.Expr
            The variable to use in the generating function.

        Returns:
        --------
        sp.Expr
            The generating function of the Hodge curve.
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

    @dispatch(int, sp.Expr)
    def _to_adams(self, degree: int, ph: sp.Expr) -> sp.Expr:
        """
        Applies Adams operations to any instances of this Hodge motive in a polynomial.

        Args:
        -----
        degree : int
            The degree of the Adams operator to apply.
        ph : sp.Expr
            The polynomial in which the Adams operator is applied.

        Returns:
        --------
        sp.Expr
            The polynomial with Adams operators applied to the Hodge motive.
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

    @dispatch(set, GrothendieckRingContext)
    def _to_adams(self, operands: set[Operand], gc: GrothendieckRingContext) -> sp.Expr:
        """
        Converts this subtree into an equivalent Adams polynomial.

        Args:
        -----
        operands : set[Operand]
            The set of all operands in the expression tree.
        gc : GrothendieckRingContext
            The Grothendieck ring context used for conversion between operators.

        Returns:
        --------
        sp.Expr
            The Adams polynomial equivalent to this subtree.
        """
        return self.get_adams_var(1)

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

    def _subs_adams(self, gc: GrothendieckRingContext, ph: sp.Expr) -> sp.Expr:
        """
        Substitutes Adams variables for equivalent Lambda polynomials in the given polynomial.

        This method is called during the `to_lambda` process to convert Adams variables 
        that appear after converting the expression tree to an Adams polynomial.

        Args:
        -----
        gc : GrothendieckRingContext
            The Grothendieck ring context used for conversion between operators.
        ph : sp.Expr
            The polynomial in which to substitute the Adams variables.

        Returns:
        --------
        sp.Expr
            The polynomial with Adams variables substituted by equivalent Lambda polynomials.
        """
        max_adams = -1
        operands = ph.free_symbols
        for i, adams in reversed(list(enumerate(self._adams_vars))):
            if adams in operands:
                max_adams = i
                break

        # Generate the lambda_to_adams polynomials up to the degree needed
        self._generate_lambda_vars(len(self._adams_vars) - 1)

        for i in range(len(self._lambda_to_adams), max_adams + 1):
            self._lambda_to_adams.append(expr_from_pol(self._lambda_to_adams_pol[i]))

        # Substitute the Adams variables into the polynomial
        ph = ph.xreplace(
            {
                self.get_adams_var(i): self._lambda_to_adams[i]
                for i in range(1, max_adams + 1)
            }
        )

        return ph
