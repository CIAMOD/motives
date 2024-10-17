import sympy as sp
from sympy.polys.rings import PolyElement
import math

from ...curves.curve import Curve

from ...lefschetz import Lefschetz

from ....core import LambdaRingContext
from ....core.lambda_ring_expr import LambdaRingExpr

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