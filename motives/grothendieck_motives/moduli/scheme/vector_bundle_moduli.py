import sympy as sp
import math

from ...curves.curve import Curve

from ...lefschetz import Lefschetz

from ....core.lambda_ring_expr import LambdaRingExpr

# TODO: Documentacion de clase VectorBundleModuli
class VectorBundleModuli:
    def __init__(self, x: Curve, p: int, r: int):
        """
        Initializes the VectorBundleModuli class with the given curve and parameters.

        Args:
        -----
        x : Curve
            The curve motive used in the formula.
        p : int
            A positive integer such that deg(L) = 2g - 2 + p.
        r : int
            The rank of the underlying vector bundles (must be 2 or 3).
        """
        if r not in [2, 3]:
            raise ValueError("The rank should be either 2 or 3")
        
        self.cur = x
        self.g = self.cur.g
        self.p = p
        self.r = r
        self.dl = 2 * self.g - 2 + p
        self.lef = Lefschetz()
        self.vhs: dict[tuple, LambdaRingExpr] = {}

        self._initiate_vector_bundles()
    def _initiate_vector_bundles(self):
        """
        Initiates the Vector Bundle Moduli for the first few dimensions.
        """

        # TODO: Falta hacer VHS[(1,)]
        self.vhs[(2,)] = (
            self.lef ** (4 * self.dl + 4 - 4 * self.g)
            * (self.cur.Jac * self.cur.P(self.lef) - self.lef**self.g * self.cur.Jac**2)
            / ((self.lef - 1) * (self.lef**2 - 1))
        )
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
        return
    
    def _calculate_vector_bundle(self, n: tuple) -> sp.Expr:
        """
        Calculates the Vector Bundle Moduli for the given tuple `(n,)`. Private method.
        """
        # TODO: Implementar el calculo del Vector Bundle de dimension `n`
        # No tenemos aun la formula para calcularlo, hay que buscarla 
        raise NotImplementedError(f"The calculation of the Vector Bundle Moduli with dimension {n} is not yet implemented.")
    
    def calculate_vector_bundle(self, n: tuple) -> sp.Expr:
        """
        Calculates the Vector Bundle Moduli for the given tuple `(n,)`. Public method.
        """
        if n not in self.vhs:
            self._calculate_vector_bundle(n)
        return self.vhs[n]
    
    def get_vector_bundle(self, n: int) -> sp.Expr:
        """
        Returns the Vector Bundle Moduli for the given int `n`.
        """
        if (n,) not in self.vhs:
            self.calculate_vector_bundle(n)
        return self.vhs[(n,)]