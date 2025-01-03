import sympy as sp
import math

from ...curves.curve import Curve

from .bundle_moduli import BundleModuli
from .vector_bundle_moduli import VectorBundleModuli



# TODO: Documentacion de clase VHS
class VHS(BundleModuli):
    def __init__(self, x: Curve, rks: tuple[int], degs:tuple[int], p:int):
        """
        Class which computes the Grothendieck motive of the moduli space of
        L-twisted Variations of Hodge Structure over the given curve x, with a twisting
        L of degree deg(L)=2g-2+p.

        Args:
        -----
        x : Curve
            The curve motive used in the formula.
        rks : tuple(int)
            The ranks of the components of the VHS.
        degs : tuple(int)
            The degrees of the components of the VHS.
        p: int
            Twisting, deg(L)=2g-2+p
        """        
        super().__init__(x)

        self.rks=rks
        self.r=sum(rks)
        if (self.r>3):
            raise NotImplementedError(f"Not implemented yet for total rank greater than 3")
        self.degs=degs
        self.d=sum(degs)
        self.p=p
        self.dl = 2 * self.g - 2 + p

        self._et_repr=self._compute_vhs(rks,degs)

    def _compute_vhs(self,rks:tuple[int], degs:tuple[int]):
        """
        Initiates the VHS for the first few dimensions.
        """
        d=sum(degs)

        if rks==(1,1):
            if self.d/2<degs[1] and degs[1] <= (self.d-self.dl)/2:
                return(
                    self.cur.Jac
                    * self.cur.lambda_(self.d-2*degs[0] + self.dl)
                )
            else:
                return sp.Integer(0)
        elif rks==(1,2):
            if self.d/3<degs[1] and degs[1] <= self.d/3-self.dl/2:
                raise NotImplementedError(f"Rank 3 not implemented yet")
            else:
                return sp.Integer(0)
        elif rks==(2,1):
            return self._compute_vhs((1,2),(degs[0]-d,-degs[1]))
        elif rks==(1,1,1):
            raise NotImplementedError(f"Rank 3 not implemented yet")
        else:
            raise NotImplementedError(f"Type not implemented yet")
        

    def _calculate_vhs(self, n: tuple) -> sp.Expr:
        """
        Calculates the VHS for the given tuple `n`. Private method.
        """
        raise NotImplementedError(
            f"The calculation of the VHS with dimension {n} is not yet implemented."
        )

    def calculate(self) -> sp.Expr:
        """
        Calculates the VHS for the given tuple. Public method.
        """
        return self._et_repr
