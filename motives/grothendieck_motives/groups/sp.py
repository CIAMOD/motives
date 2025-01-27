from .general_groups import C
from typing import Tuple


# TODO docs


class SP(C):
    """
    Represents the Grothendieck motive of the complex symplectic group Sp(2n, C).

    This class inherits from the C class and represents the symplectic group Sp(2n, C),
    which consists of 2n x 2n matrices preserving a symplectic form.

    Attributes
    ----------
    n : int
        The dimension of the symplectic group SP(n).

    Methods
    -------
    __new__(cls, n: int, *args, **kwargs) -> "SP":
        Creates a new instance of the SP class with the specified dimension n.
    __repr__(self) -> str:
        Returns a string representation of the SP group.
    _hashable_content(self) -> Tuple[int]:
        Returns a tuple containing the dimension n, used for hashing.
    """

    def __new__(cls, n: int, *args, **kwargs) -> "SP":
        """
        Creates a new instance of the SP class with the specified dimension n.

        Parameters
        ----------
        n : int
            The dimension of the symplectic group SP(n).
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        SP
            A new instance of the SP class.

        Raises
        ------
        ValueError
            If n is not an even integer.
        """
        if n % 2 != 0:
            raise ValueError(
                "The dimension of the symplectic group SP(n) must be even."
            )

        new_sl = C.__new__(cls, n // 2)
        new_sl.n = n
        return new_sl

    def __repr__(self) -> str:
        """
        Returns a string representation of the SP group.

        Returns
        -------
        str
            A string representation of the SP group in the format 'SP_n'.
        """
        return f"SP_{self.n}"

    def _hashable_content(self) -> Tuple[int]:
        """
        Returns a tuple containing the dimension n, used for hashing.

        Returns
        -------
        tuple of int
            A tuple containing the dimension n.
        """
        return (self.n,)
