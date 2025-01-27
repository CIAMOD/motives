import sympy as sp
import math

from ...curves.curve import Curve
from .bundle_moduli import BundleModuli


class VHS(BundleModuli):
    def __init__(self, x: Curve, rks: tuple, degs: tuple, p: int):
        """
        Computes the Grothendieck motive of the moduli space of
        L-twisted Variations of Hodge Structure over the given curve `x`.

        Args:
            x (Curve): The curve motive used in the formula.
            rks (tuple): The ranks of the components of the VHS.
            degs (tuple): The degrees of the components of the VHS.
            p (int): Twisting parameter, deg(L) = 2g - 2 + p.
        """
        super().__init__(x)

        self.rks = rks
        self.r = sum(rks)
        if self.r > 3:
            raise NotImplementedError(
                "Not implemented yet for total rank greater than 3"
            )
        self.degs = degs
        self.d = sum(degs)
        self.dl = 2 * self.g - 2 + p

        self._et_repr = self._compute_vhs(rks, degs)

    def _compute_vhs(self, rks: tuple, degs: tuple) -> sp.Expr:
        """
        Initializes the VHS for the first few dimensions.

        Args:
            rks (tuple): The ranks of the components of the VHS.
            degs (tuple): The degrees of the components of the VHS.

        Returns:
            sp.Expr: The computed VHS expression.
        """
        d1 = degs[0]

        if rks == (1, 1):
            if self.d / 2 < d1 <= (self.d + self.dl) / 2:
                return self.cur.Jac * self.cur.lambda_(self.d - 2 * d1 + self.dl)
            else:
                return sp.Integer(0)

        elif rks == (1, 2):
            if self.d / 3 < d1 < self.d / 3 + self.dl / 2:
                return (
                    self.cur.Jac**2
                    / (self.lef - 1)
                    * (
                        self.lef
                        ** (2 * math.floor(self.d / 3) - self.d + d1 + self.g + 1)
                        * (self.cur + self.lef**2).lambda_(
                            self.d - math.floor(self.d / 3) - 2 * d1 + self.dl - 1
                        )
                        - (self.lef * self.cur + 1).lambda_(
                            self.d - math.floor(self.d / 3) - 2 * d1 + self.dl - 1
                        )
                    )
                )
            else:
                return sp.Integer(0)

        elif rks == (2, 1):
            return self._compute_vhs((1, 2), (d1 - self.d, -d1))

        elif rks == (1, 1, 1):
            d2 = degs[1]

            if math.ceil(self.d / 3) <= d1 <= self.dl + math.floor(self.d / 3) and max(
                -self.dl + d1, math.ceil(2 * self.d / 3) - d1
            ) <= d2 <= math.floor((self.d + self.dl - d1) / 2):
                return (
                    self.cur.Jac
                    * self.cur.lambda_(-d1 + d2 + self.dl)
                    * self.cur.lambda_(self.d - d1 - 2 * d2 + self.dl)
                )

        else:
            raise NotImplementedError("Type not implemented yet")

    def _calculate_vhs(self, n: tuple) -> sp.Expr:
        """
        Calculates the VHS for the given tuple `n`. Not implemented.

        Args:
            n (tuple): The parameters for VHS calculation.

        Raises:
            NotImplementedError: Always raised as the method is not implemented.
        """
        raise NotImplementedError(
            f"The calculation of the VHS with dimension {n} is not yet implemented."
        )

    def get_max_adams_degree(self) -> int:
        """
        Returns the maximum degree of the Adams operator for this VHS.

        Returns:
            int: The maximum Adams degree.
        """
        return self._et_repr.get_max_adams_degree()

    def calculate(self) -> sp.Expr:
        """
        Calculates the VHS expression.

        Returns:
            sp.Expr: The computed VHS.
        """
        return self._et_repr
