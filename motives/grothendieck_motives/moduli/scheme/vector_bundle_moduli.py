import sympy as sp
import math

from ...curves.curve import Curve

from ....utils import partitions, prod, sp_decimal_part
from .bundle_moduli import BundleModuli


class VectorBundleModuli(BundleModuli):
    def __init__(self, x: Curve, r: int, d: int):
        """
        Class for computing the motive of the moduli space of vector bundles
        of rank r and degree d over a curve x.

        Args:
        -----
        x : Curve
            The curve motive used in the formula.
        r : int
            The rank of the vector bundles.
        d : int
            The degree of the vector bundles.
        """

        super().__init__(x)
        self.r = r
        self.d = d

        self._initiate_vector_bundle_moduli()

    def _initiate_vector_bundle_moduli(self):
        """
        Initiates the Vector Bundle Moduli for the first few dimensions.
        """
        if self.r == 1:
            self._et_repr: sp.Expr = self.cur.Jac
        elif self.r == 2 and self.d % 2 != 0:
            self._et_repr: sp.Expr = self._compute_motive_rk2()
        elif self.r == 3 and self.d % 3 != 0:
            self._et_repr: sp.Expr = self._compute_motive_rk3()
        else:
            self._et_repr: sp.Expr = self._compute_motive_rkr(self.r, self.d)

    def _compute_motive_rk2(self) -> sp.Expr:
        """
        Calculates the expresion of the motive of the moduli space of vector bundles in rank 2
        using the equation from [Sánchez '14]
        """
        return (
            self.cur.Jac * self.cur.P(self.lef) - self.lef**self.g * self.cur.Jac**2
        ) / ((self.lef - 1) * (self.lef**2 - 1))

    def _compute_motive_rk3(self) -> sp.Expr:
        """
        Calculates the expresion of the motive of the moduli space of vector bundles in rank 3
        using the equation from [García-Prada, Heinloth, Schmitt '14]
        """
        return (
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

    def _compute_motive_rkr(self, r: int, d: int) -> sp.Expr:
        """
        Calculates the expresion of the motive of the moduli space of vector bundles in rank r
        using the general equation from [del Baño '01].

        Args:
        -----
        r : int
            The rank of the vector bundles.
        d : int
            The degree of the vector bundles.
        """
        return sum(
            (-1) ** (s - 1)
            * self.cur.P(1) ** s
            / (1 - self.lef) ** (s - 1)
            * prod(
                (self.cur.Z(self.lef**i) for j in range(s) for i in range(1, part[j]))
            )
            * prod(1 / (1 - self.lef ** (part[j] + part[j + 1])) for j in range(s - 1))
            * self.lef
            ** (
                sum(
                    part[i] * part[j] * (self.cur.g - 1)
                    for j in range(s)
                    for i in range(j)
                )
                + sum(
                    (part[i] + part[i + 1])
                    * sp_decimal_part(-sum(part[k] for k in range(i + 1)) * d, r)
                    for i in range(s - 1)
                )
            )
            for s in range(1, r + 1)
            for part in partitions(r, s, minimum=1)
        )

    def calculate(self) -> sp.Expr:
        """
        Calculates the vector bundle moduli for the given tuple `(n,)`. Public method.
        """
        return self._et_repr
