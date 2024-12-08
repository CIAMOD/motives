import sympy as sp
import math

from ...curves.curve import Curve

from .vector_bundle_moduli import VectorBundleModuli


# TODO: Documentacion de clase VHS
class VHS(VectorBundleModuli):
    def __init__(self, x: Curve, p: int, r: int):
        """
        Initializes the VHS class with the given curve and parameters.

        Args:
        -----
        x : Curve
            The curve motive used in the formula.
        p : int
            A positive integer such that deg(L) = 2g - 2 + p.
        r : int
            The rank of the underlying vector bundles (must be 2 or 3).
        """
        if r not in [1, 2, 3]:
            raise ValueError("The rank should be either 2 or 3")

        super().__init__(x, p, r)
        self._initiate_vhs()

    def _initiate_vhs(self):
        """
        Initiates the VHS for the first few dimensions.
        """

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

    def _calculate_vhs(self, n: tuple) -> sp.Expr:
        """
        Calculates the VHS for the given tuple `n`. Private method.
        """
        raise NotImplementedError(
            f"The calculation of the VHS with dimension {n} is not yet implemented."
        )

    def calculate_vhs(self, n: tuple) -> sp.Expr:
        """
        Calculates the VHS for the given tuple `n`. Public method.
        """
        if n not in self.vhs:
            if len(n) == 1:
                self.calculate_vector_bundle(n)
            else:
                self._calculate_vhs(n)
        return self.vhs[n]

    def get_vhs(self, n: tuple) -> sp.Expr:
        """
        Returns the VHS for the given tuple `n`.
        """
        if n not in self.vhs:
            self.calculate_vhs(n)
        return self.vhs[n]
