import sympy as sp
from sympy.polys.rings import PolyElement
import math

from ..curve import Curve

from ...lefschetz import Lefschetz
from ...symbol import Symbol

from ....core import LambdaRingContext
from ....core.lambda_ring_expr import LambdaRingExpr
from ....utils import all_partitions, expand_variable, expand_variable_1mtk, expand_variable_1mt, subs_variable, Partitions

class TwistedHiggsModuliBB:
    """
    The motive of the moduli space of L-twisted Higgs bundles over the curve X.

    This class computes the motive of the moduli space of L-twisted Higgs bundles of rank r
    over the algebraic curve X. It uses the formula derived from the Bialynicki-Birula decomposition
    of the moduli space, as described in [Alfaya, Oliveira 2024, Corollary 8.1].

    Attributes:
    -----------
    cur : Curve
        The motive of an algebraic curve of genus g >= 2.
    g : int
        Positive integer representing the genus of the curve.
    p : int
        Positive integer such that deg(L) = 2g - 2 + p.
    r : int
        Rank of the underlying vector bundles (currently implemented for r <= 3).
    dl : int
        Degree of the line bundle, calculated as 2g - 2 + p.
    lef : Lefschetz
        The Lefschetz operator.
    vhs : dict[tuple, LambdaRingExpr]
        A dictionary storing the different vhs components of the formula.
    """

    def __init__(self, x: Curve, p: int, r: int):
        """
        Initializes the TwistedHiggsModuliBB class with the given curve and parameters.

        Args:
        -----
        x : Curve
            The curve motive used in the formula.
        p : int
            A positive integer such that deg(L) = 2g - 2 + p.
        r : int
            The rank of the underlying vector bundles (must be 2 or 3).
        """
        self.cur = x
        self.g = self.cur.g
        self.p = p
        self.r = r
        self.dl = 2 * self.g - 2 + p
        self.lef = Lefschetz()
        self.vhs: dict[tuple, LambdaRingExpr] = {}

        if self.r == 2:
            self.M2()
        elif self.r == 3:
            self.M3()
        else:
            raise ValueError("The rank should be either 2 or 3")

    def M2(self) -> None:
        """
        Computes the formula of rank 2 and fills the vhs dictionary with the components.

        This method calculates and stores the formula for rank 2 Higgs bundles in the vhs dictionary.
        """
        self.vhs[(2,)] = (
            self.lef ** (4 * self.dl + 4 - 4 * self.g)
            * (self.cur.Jac * self.cur.P(self.lef) - self.lef**self.g * self.cur.Jac**2)
            / ((self.lef - 1) * (self.lef**2 - 1))
        )
        self.vhs[(1, 1)] = (
            self.lef ** (3 * self.dl + 2 - 2 * self.g)
            * self.cur.Jac
            * sp.Add(
                *[
                    self.cur.lambda_(1 - 2 * i + self.dl)
                    for i in range(1, (self.dl + 1) // 2 + 1)
                ]
            )
        )

    def M3(self) -> None:
        """
        Computes the formula of rank 3 and fills the vhs dictionary with the components.

        This method calculates and stores the formula for rank 3 Higgs bundles in the vhs dictionary.
        """
        self.vhs[(3,)] = (
            self.lef ** (9 * self.dl + 9 - 9 * self.g)
            * self.cur.Jac
            * (
                self.lef ** (3 * self.g - 1)
                * (1 + self.lef + self.lef**2)
                * self.cur.Jac**2
                - self.lef ** (2 * self.g - 1)
                * (1 + self.lef) ** 2
                * self.cur.Jac
                * self.cur.P(self.lef)
                + self.cur.P(self.lef) * self.cur.P(self.lef**2)
            )
            / ((self.lef - 1) * (self.lef**2 - 1) ** 2 * (self.lef**3 - 1))
        )
        self.vhs[(1, 2)] = (
            (self.lef ** (7 * self.dl + 5 - 5 * self.g) * self.cur.Jac**2)
            * sp.Add(
                *[
                    self.lef ** (i + self.g)
                    * sp.Add(
                        *[
                            self.cur.lambda_(j)
                            * self.lef ** (2 * (-2 * i + self.dl - j))
                            for j in range(-2 * i + self.dl + 1)
                        ]
                    )
                    - sp.Add(
                        *[
                            self.cur.lambda_(j) * self.lef**j
                            for j in range(-2 * i + self.dl + 1)
                        ]
                    )
                    for i in range(1, math.floor(self.dl / 2 + 1 / 3) + 1)
                ]
            )
            / (self.lef - 1)
        )
        self.vhs[(2, 1)] = (
            (self.lef ** (7 * self.dl + 5 - 5 * self.g) * self.cur.Jac**2)
            * sp.Add(
                *[
                    self.lef ** (i + self.g - 1)
                    * sp.Add(
                        *[
                            self.cur.lambda_(j)
                            * self.lef ** (2 * (-2 * i + self.dl + 1 - j))
                            for j in range(-2 * i + self.dl + 1 + 1)
                        ]
                    )
                    - sp.Add(
                        *[
                            self.cur.lambda_(j) * self.lef**j
                            for j in range(-2 * i + self.dl + 1 + 1)
                        ]
                    )
                    for i in range(1, math.floor(self.dl / 2 + 2 / 3) + 1)
                ]
            )
            / (self.lef - 1)
        )
        self.vhs[(1, 1, 1)] = (
            self.lef ** (6 * self.dl + 3 - 3 * self.g)
            * self.cur.Jac
            * sp.Add(
                *[
                    self.cur.lambda_(-i + j + self.dl)
                    * self.cur.lambda_(1 - i - 2 * j + self.dl)
                    for i in range(1, self.dl + 1)
                    for j in range(
                        max(-self.dl + i, 1 - i), math.floor((self.dl - i + 1) / 2) + 1
                    )
                ]
            )
        )

    def simplify(
        self, lrc: LambdaRingContext = None, *, verbose: int = 0
    ) -> PolyElement:
        """
        Transforms the equation into a polynomial of lambdas and simplifies it.

        This method transforms the vhs dictionary into a polynomial of lambda operators,
        cancels any denominators, and simplifies the result. It can print intermediate steps
        depending on the verbosity level.

        Args:
        -----
        lrc : LambdaRingContext, optional
            The Grothendieck ring context to use for the simplification.
        verbose : int, optional
            How much information to print (0 for none, 1 for progress, 2 for intermediate formulas).

        Returns:
        --------
        PolyElement
            The simplified motive as a polynomial.
        """
        if verbose > 0:
            print(
                f"Simplifying the formula of g={self.g}, p={self.p}, r={self.r} with BB"
            )

        result: sp.Poly = 0
        domain_qq = sp.QQ[[self.lef] + self.cur.curve_hodge.lambda_symbols[1:]]
        domain_zz = sp.ZZ[[self.lef] + self.cur.curve_hodge.lambda_symbols[1:]]

        # Initialize a Grothendieck ring context if not provided
        lrc = lrc or LambdaRingContext()

        # Process each VHS component and simplify
        for vhs, expr in self.vhs.items():
            # Convert to lambda variables
            lambda_pol = expr.to_lambda(lrc=lrc)

            if verbose > 0:
                print(f"Computed lambda of VHS {vhs}")
                if verbose > 1:
                    print(f"Lambda of VHS {vhs}: {lambda_pol}")

            if all(el == 1 for el in vhs):
                result += domain_qq.from_sympy(lambda_pol)
                if verbose > 0:
                    print(f"Cancelled VHS {vhs}")
                continue

            # Cancel the denominator
            numerator, denominator = sp.fraction(lambda_pol)
            numerator_pol = domain_qq.from_sympy(numerator)
            denominator_pol = domain_qq.from_sympy(denominator)
            numerator_pol = numerator_pol.exquo(denominator_pol)

            result += numerator_pol

            if verbose > 0:
                print(f"Cancelled VHS {vhs}")
                if verbose > 1:
                    print(f"Cancelled VHS {vhs}: {numerator_pol}")

        return domain_zz.convert_from(result, domain_qq)


class TwistedHiggsModuliADHM:
    """
    The motive of the moduli space of L-twisted Higgs bundles over the curve X.

    This class computes the motive of the moduli space of L-twisted Higgs bundles of rank r
    over the algebraic curve X. It uses the conjectural equation from [Mozgovoy 2012, Conjecture 3]
    as a solution to the ADHM equation, based on [Chuang, Diaconescu, Pan 2011].

    Attributes:
    -----------
    cur : Curve
        The motive of an algebraic curve of genus g >= 2.
    g : int
        The genus of the curve.
    p : int
        A positive integer such that deg(L) = 2g - 2 + p.
    r : int
        The rank of the underlying vector bundles.
    lef : Lefschetz
        The Lefschetz operator.
    t : Symbol
        The symbol t that will be substituted for 1 during calculations.
    T : Symbol
        The symbol T used in the generating function.
    gen_f : LambdaRingExpr
        The generating function used to compute the Mozgovoy formula.
    coeff : dict[sp.Expr, sp.Expr]
        A dictionary storing the coefficients of the generating function.
    """

    def __init__(self, x: Curve, p: int, r: int):
        """
        Initializes the TwistedHiggsModuliADHM class with the given curve and parameters.

        Args:
        -----
        x : Curve
            The curve motive used in the formula.
        p : int
            The number of points of the moduli (deg(L) = 2g - 2 + p).
        r : int
            The rank of the underlying vector bundles.
        """
        self.cur = x
        self.g = self.cur.g
        self.p = p
        self.r = r
        self._mobius = [1, -1, -1, 0, -1, 1, -1, 0, 0, 1, -1, 0, -1, 1, 1, 0]
        self.lef = Lefschetz()
        self.t, self.T = Symbol("t"), Symbol("T")
        self.gen_f: LambdaRingExpr = 0
        self.coeff: dict[sp.Expr, sp.Expr] = {}

    def gen_func(self) -> LambdaRingExpr:
        """
        Generates the function used to compute the Mozgovoy formula.

        This method constructs the generating function up to rank r using the ADHM equation.

        Returns:
        --------
        LambdaRingExpr
            The generating function up to rank r.
        """
        H_sum: list[LambdaRingExpr] = [0] * (self.r + 1)
        for j in range(1, self.r + 1):
            Hn: LambdaRingExpr = 0
            for partition in all_partitions(j):
                part = Partitions(partition)
                Hp = 1
                for el in part.elements:
                    a = part.a(*el)
                    l = part.l(*el)
                    h = a + l + 1
                    Hp *= (
                        (-(self.t ** (a - l)) * self.lef**a) ** self.p
                        * self.t ** ((1 - self.g) * (2 * l + 1))
                        * self.cur.Z(self.t**h * self.lef**a)
                    )
                Hn += Hp
            H_sum[j] = Hn

        for j in range(1, self.r + 1):
            for k in range(1, self.r + 1):
                pleth = 0
                for h in range(1, self.r // j + 1 - (k - 1)):
                    pleth += H_sum[h] * self.T**h
                if isinstance(pleth, int):
                    continue
                pleth = pleth.adams(j)

                self.gen_f += (
                    sp.Integer((-1) ** (k + 1) * self._mobius[j - 1])
                    / sp.Integer(j * k)
                    * pleth**k
                )

        return self.gen_f

    def moz_lambdas(
        self,
        lrc: LambdaRingContext = None,
        *,
        verbose: int = 0,
    ) -> PolyElement:
        """
        Computes the motive of rank r as a polynomial of lambda operators.

        This method derives the formula for the motive of rank r and converts it into a
        polynomial of lambda operators using the ADHM conjecture.

        Args:
        -----
        lrc : LambdaRingContext, optional
            The Grothendieck ring context to use for the conversion.
        verbose : int, optional
            How much information to print during the computation (0 for none, 1 for progress, 2 for formulas).

        Returns:
        --------
        PolyElement
            The motive of rank r, turned into a polynomial of lambdas.
        """
        if verbose > 0:
            print(f"Computing the formula of g={self.g}, p={self.p}, r={self.r} with ADHM")

        if self.gen_f == 0:
            self.gen_func()

        if verbose > 0:
            print("Formula computed.")

        t, T = Symbol("t"), Symbol("T")
        lef = Lefschetz()

        if self.coeff == {}:
            lrc = lrc or LambdaRingContext()
            sp_M_sum = self.gen_f.to_lambda(lrc=lrc)

            if verbose > 0:
                print("Formula converted to lambdas.")
                print("Max adams degree:", self.gen_f.get_max_adams_degree())
                if verbose > 1:
                    print(sp_M_sum)

            # Prepare the expression for collecting terms with the variable T
            sp_M_sum_expanded = sp_M_sum.replace(*expand_variable(T))
            if verbose > 0:
                print("Formula expanded.")
                if verbose > 1:
                    print(sp_M_sum_expanded)

            self.coeff = sp.collect(sp_M_sum_expanded, T, evaluate=False)
            if verbose > 0:
                print("Coefficients computed.")

        # Multiply terms for easier collection of 1 / (1 - t)
        Hr: sp.Expr = sp.Add(
            *[(1 - t) * (1 - lef * t) * term for term in self.coeff[T**self.r].args]
        )

        # Expand (1 - t ** k) into (1 - t) * (1 + t + ... + t ** (k - 1))
        Hr = Hr.replace(*expand_variable_1mtk(t))

        # Prepare the expression for collection of 1 / (1 - t)
        Hr = Hr.replace(*expand_variable_1mt(t))
        if verbose > 0:
            print("Formula prepared for collection in 1 / (1 - t).")

        poly_domain = sp.ZZ[self.cur.curve_hodge.lambda_symbols[1:]]
        domain = sp.ZZ[[lef] + self.cur.curve_hodge.lambda_symbols[1:]]

        # Cancel the necessary terms and substitute
        Hr = subs_variable(Hr, t, 1, lef, domain=poly_domain, verbose=verbose)

        et_lambda = (
            domain.from_sympy(
                (-1) ** (self.p * self.r)
                * lef
                ** (self.r**2 * (self.g - 1) + self.p * self.r * (self.r + 1) // 2)
            )
            * Hr
        )

        if verbose > 0:
            print("Final formula computed")
            if verbose > 1:
                print(et_lambda)

        return et_lambda


class TwistedHiggsModuli:
    """
    The motive of the moduli space of L-twisted Higgs bundles over the curve X.

    This class computes the motive of the moduli space of L-twisted Higgs bundles of rank r
    over the algebraic curve X in the completion of the Grothendieck ring of Chow motives over C.
    By default, it uses the BB derivation for r <= 3 and the ADHM derivation for r > 3.

    Attributes:
    -----------
    cur : Curve
        The curve motive used in the formula.
    g : int
        The genus of the curve.
    p : int
        Positive integer representing the degree of L (deg(L) = 2g - 2 + p).
    r : int
        The rank of the underlying vector bundles.
    method : str
        The method used to derive the formula ('BB', 'ADHM', or 'auto').
    motive : TwistedHiggsModuliBB or TwistedHiggsModuliADHM
        The object representing the motive of the moduli space using either the BB or ADHM method.
    """

    def __init__(self, x: Curve, p: int, r: int, method: str = "auto"):
        """
        Initializes the TwistedHiggsModuli class with the given curve and parameters.

        Args:
        -----
        x : Curve
            The curve motive used in the formula.
        p : int
            The number of points of the moduli space (deg(L) = 2g - 2 + p).
        r : int
            The rank of the underlying vector bundles.
        method : str, optional
            The method to use to derive the formula. By default ('auto'), it uses the BB derivation for
            r <= 3 and the ADHM derivation for r > 3. It can also be explicitly set to 'BB' or 'ADHM'.
        """
        self.cur = x
        self.g = self.cur.g
        self.p = p
        self.r = r

        # Automatically decide which method to use if 'auto' is selected
        if method == "auto":
            if r <= 3:
                self.method = "BB"
            else:
                self.method = "ADHM"
        else:
            self.method = method

        # Initialize the motive using either BB or ADHM methods
        if self.method == "BB":
            self.motive = TwistedHiggsModuliBB(x, p, r)
        elif self.method == "ADHM":
            self.motive = TwistedHiggsModuliADHM(x, p, r)
        else:
            raise ValueError("The method should be either 'BB', 'ADHM', or 'auto'")

    def compute(
        self, lrc: LambdaRingContext = None, *, verbose: int = 0
    ) -> PolyElement:
        """
        Computes the motive of the moduli space of L-twisted Higgs bundles.

        This method derives the formula for the moduli space of L-twisted Higgs bundles
        using the selected method (either BB or ADHM) and computes the final motive.

        Args:
        -----
        lrc : LambdaRingContext, optional
            The Grothendieck ring context to use for the computation.
        verbose : int, optional
            How much information to print during the computation (0 for none, 1 for progress, 2 for formulas).

        Returns:
        --------
        PolyElement
            The motive of the moduli space of L-twisted Higgs bundles, represented as a polynomial.
        """
        if self.method == "BB":
            return self.motive.simplify(lrc=lrc, verbose=verbose)
        else:
            return self.motive.moz_lambdas(lrc=lrc, verbose=verbose)
