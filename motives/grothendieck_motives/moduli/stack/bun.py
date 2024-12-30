from .bg import BunG
from ...curves.curve import Curve
from ...groups.gl import GL


class Bun(BunG):
    """
    Represents the moduli stack of vector bundles on a curve with a general linear group GL(n).

    The `Bun` is a specialized version of `BunG` where the group is specifically GL(n).
    It supports Adams and Lambda operations, generating functions, and interacts with other motives.

    Attributes:
    -----------
    curve : Curve
        The curve for which this Bun is defined.
    n : int
        The dimension of the general linear group GL(n).
    """

    n: int

    def __new__(cls, curve: Curve, n: int, *args, **kwargs):
        """
        Creates a new instance of the `Bun` class.

        Args:
        -----
        curve : Curve
            The curve for which to create the Bun.
        n: int
            The dimension of the general linear group GL(n).

        Returns:
        --------
        Bun
            A new instance of the `Bun` class.
        """
        new_bun = BunG.__new__(cls, curve, GL(n))
        new_bun.n = n
        return new_bun

    def __init__(self, curve: Curve, n: int, *args, **kwargs):
        """
        TODO
        """
        super().__init__(curve, GL(n))

    def __repr__(self) -> str:
        """
        Returns the string representation of the Bun.

        Returns:
        --------
        str
            A string representation in the form of "Bun_{curve}_{n}".
        """
        return f"Bun_{self.curve}_{self.n}"
