from sympy.polys.rings import PolyElement

from ...curves.curve import Curve

from .bundle_moduli import BundleModuli
from .twisted_higgs_moduli_bb import TwistedHiggsModuliBB
from .twisted_higgs_moduli_adhm import TwistedHiggsModuliADHM


class TwistedHiggsModuli(BundleModuli):
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

        self.p = p
        self.r = r
        super().__init__(x)
        
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

    def compute(self, *, verbose: int = 0) -> PolyElement:
        """
        Computes the motive of the moduli space of L-twisted Higgs bundles.

        This method derives the formula for the moduli space of L-twisted Higgs bundles
        using the selected method (either BB or ADHM) and computes the final motive.

        Args:
        -----
        verbose : int, optional
            How much information to print during the computation (0 for none, 1 for progress, 2 for formulas).

        Returns:
        --------
        PolyElement
            The motive of the moduli space of L-twisted Higgs bundles, represented as a polynomial.
        """
        if self.method == "BB":
            return self.motive.simplify(verbose=verbose)
        else:
            return self.motive.moz_lambdas(verbose=verbose)
