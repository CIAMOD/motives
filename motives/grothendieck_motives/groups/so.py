from .general_groups import B, D


# TODO docs


class SO(B):
    """
    Represents the special orthogonal group SO(n) as a Grothendieck motive.
    This class inherits from the G class and represents the special orthogonal group SO(n),
    which consists of n x n orthogonal matrices with determinant 1. The motive is constructed
    using a combination of even integers from 2 to 2n-2 and the integer n, with the dimension n * (2n - 1).

    Attributes:
    -----------
    n : int
        The dimension of the special orthogonal group SO(n).

    Methods:
    --------
    __new__(cls, n: int, *args, **kwargs)
        Creates a new instance of the SO class with the specified dimension n.
    __repr__() -> str
        Returns a string representation of the SO group.
    _hashable_content() -> tuple
        Returns a tuple containing the dimension n, used for hashing.
    """

    def __new__(cls, n: int, *args, **kwargs):
        """
        Creates a new instance of the SO class with the specified dimension n.

        Parameters:
        -----------
        n : int
            The dimension of the special orthogonal group SO(n).
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns:
        --------
        SO
            A new instance of the SO class.
        """
        if n % 2 == 1:
            new_sl = B.__new__(cls, (n - 1) // 2)
        elif n % 2 == 0:
            new_sl = D.__new__(cls, n // 2)

        new_sl.n = n
        return new_sl

    def __repr__(self) -> str:
        """
        Returns a string representation of the SO group.

        Returns:
        --------
        str
            A string representation of the SO group in the format 'SO_n'.
        """
        return f"SO_{self.n}"

    def _hashable_content(self) -> tuple:
        """
        Returns a tuple containing the dimension n, used for hashing.

        Returns:
        --------
        tuple
            A tuple containing the dimension n.
        """
        return (self.n,)
