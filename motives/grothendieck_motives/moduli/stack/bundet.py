from .bung import BunG
from ...curves.curve import Curve
from ...groups.sl import SL


class BunDet(BunG):
    """
    Represents the moduli stack of SL(n)-bundles with fixed determinant on a smooth complex projective curve.

    Inherits from `BunG` and specializes it for the SL(n) group. Supports Adams and Lambda operations,
    generating functions, and interacts with other motives.

    Attributes:
    -----------
    curve : Curve
        The curve for which the moduli stack is defined.
    n : int
        The rank of the vector bundle.
    """

    n: int

    def __new__(cls, curve: Curve, n: int, *args, **kwargs) -> "BunDet":
        """
        Creates a new instance of the `BunDet` class.

        Parameters:
        -----------
        curve : Curve
            The curve for which to create the BunDet.
        n : int
            The rank of the vector bundles.

        Returns:
        --------
        BunDet
            A new instance of the `BunDet` class.
        """
        new_bun = BunG.__new__(cls, curve, SL(n))
        new_bun.n = n
        return new_bun

    def __init__(self, curve: Curve, n: int, *args, **kwargs) -> None:
        """
        Initializes a `BunDet` instance.

        Parameters:
        -----------
        curve : Curve
            The curve for which the moduli stack is defined.
        n : int
            The rank of the vector bundles.
        """
        super().__init__(curve, SL(n))

    def __repr__(self) -> str:
        """
        Returns the string representation of the BunDet.

        Returns:
        --------
        str
            A string representation in the form "Bun_{curve}_{n}^det".
        """
        return f"Bun_{self.curve}_{self.n}^det"
