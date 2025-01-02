import sympy as sp
import math

from ...curves.curve import Curve

from ...lefschetz import Lefschetz

from ....core.lambda_ring_expr import LambdaRingExpr
from .bundle_moduli import BundleModuli

# TODO: Documentacion de clase VectorBundleModuli
class VectorBundleModuli(BundleModuli):
    def __init__(self, x: Curve, r: int):
        """
        Class for computing the motive of the moduli space of vector bundles
        of rank r over a curve x.

        Args:
        -----
        x : Curve
            The curve motive used in the formula.
        r : int
            The rank of the vector bundles.
        """

        super().__init__(x)
        self.r = r

        self._initiate_vector_bundle()

    def _initiate_vector_bundle(self):
        """
        Initiates the Vector Bundle Moduli for the first few dimensions.
        """
        if self.r==1:
            self._et_repr: sp.Expr= self.cur.Jac
        elif self.r==2:
            self._et_repr: sp.Expr=(
                (self.cur.Jac * self.cur.P(self.lef) - self.lef**self.g * self.cur.Jac**2)
                / ((self.lef - 1) * (self.lef**2 - 1))
            )
        elif self.r==3:
            self._et_repr= (
                self.cur.Jac
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
        else:
            raise NotImplementedError(
                f"The calculation of the Vector Bundle Moduli with dimension {self.r} is not yet implemented."
            )
        return

    def calculate(self) -> sp.Expr:
        """
        Calculates the vector bundle moduli for the given tuple `(n,)`. Public method.
        """
        return self._et_repr
