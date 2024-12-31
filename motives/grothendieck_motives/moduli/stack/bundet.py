from .bung import BunG
from ...curves.curve import Curve
from ...groups.sl import SL


class BunDet(BunG):
    """
    Represents the moduli stack of vector bundles with fixed determinant (SL(n)-bundles) on a smooth complex projective curve.

    The `Bun` is a specialized version of `BunG` where the group is specifically SL(n).
    It supports Adams and Lambda operations, generating functions, and interacts with other motives.

    Attributes:
    -----------
    curve : Curve
        The curve for which the moduli stack is defined.
    n : int
        The rank of the vector bundle.
    """

    n: int

    def __new__(cls, curve: Curve, n: int, *args, **kwargs):
        """
        Creates a new instance of the `BunDet` class.

        Args:
        -----
        curve : Curve
            The curve for which to create the BunDet.
        n: int
            The rank of the vector bundles.

        Returns:
        --------
        Bun
            A new instance of the `Bun` class.
        """
        new_bun = BunG.__new__(cls, curve, SL(n))
        new_bun.n = n
        return new_bun

    def __init__(self, curve: Curve, n: int, *args, **kwargs):
        """
        TODO
        """
        super().__init__(curve, SL(n))

    def __repr__(self) -> str:
        """
        Returns the string representation of the Bun.

        Returns:
        --------
        str
            A string representation in the form of "Bun_{curve}_{n}".
        """
        return f"Bun_{self.curve}_{self.n}^det"
