import sympy as sp
from sympy.polys.rings import PolyElement
import math

from .vhs import VHS

from ...curves.curve import Curve

from ....core import LambdaRingContext

class TwistedHiggsModuliBB(VHS):
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
        # TODO: Implement r = 1, it should be the jacobian of the curve
        if r not in [2, 3]:
            raise ValueError("The rank should be either 2 or 3")
        
        super().__init__(x, p, r)

        print("TwistedHiggsModuliBB initialized")

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

        # Choose VHS components base on rank
        if self.r == 2:
            vhs_keys = [(2,), (1, 1)]
        elif self.r == 3:
            vhs_keys = [(3,), (1, 2), (2, 1), (1, 1, 1)]  
        else:
            raise NotImplementedError("The rank should be either 2 or 3")

        # Process each VHS component and simplify
        for vhs_key in vhs_keys:
            # Access the expression associated with the current key
            expr = self.vhs.get(vhs_key)
            # Convert to lambda variables
            lambda_pol = expr.to_lambda(lrc=lrc)

            if verbose > 0:
                print(f"Computed lambda of VHS {vhs_key}")
                if verbose > 1:
                    print(f"Lambda of VHS {vhs_key}: {lambda_pol}")

            if all(el == 1 for el in vhs_key):
                result += domain_qq.from_sympy(lambda_pol)
                if verbose > 0:
                    print(f"Cancelled VHS {vhs_key}")
                continue

            # Cancel the denominator
            numerator, denominator = sp.fraction(lambda_pol)
            numerator_pol = domain_qq.from_sympy(numerator)
            denominator_pol = domain_qq.from_sympy(denominator)
            numerator_pol = numerator_pol.exquo(denominator_pol)

            result += numerator_pol

            if verbose > 0:
                print(f"Cancelled VHS {vhs_key}")
                if verbose > 1:
                    print(f"Cancelled VHS {vhs_key}: {numerator_pol}")

        return domain_zz.convert_from(result, domain_qq)