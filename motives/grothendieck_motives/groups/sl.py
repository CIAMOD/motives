from .g import G


class SL(G):
    """
    Represents the special linear group SL(n) as a Grothendieck motive.
    This class inherits from the G class and represents the special linear group SL(n),
    which consists of n x n matrices with determinant 1. The motive is constructed
    using the range of integers from 2 to n (inclusive) and the dimension n^2 - 1.

    Attributes:
    -----------
    n : int
        The dimension of the special linear group SL(n).

    Methods:
    --------
    __new__(cls, n: int, *args, **kwargs)
        Creates a new instance of the SL class with the specified dimension n.
    __repr__() -> str
        Returns a string representation of the SL group.
    _hashable_content() -> tuple
        Returns a tuple containing the dimension n, used for hashing.
    """

    def __new__(cls, n: int, *args, **kwargs):
        """
        Creates a new instance of the SL class with the specified dimension n.

        Parameters:
        -----------
        n : int
            The dimension of the special linear group SL(n).
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns:
        --------
        SL
            A new instance of the SL class.
        """
        new_sl = G.__new__(cls, range(2, n + 1), n**2 - 1)
        new_sl.n = n
        return new_sl

    def __repr__(self) -> str:
        """
        Returns a string representation of the SL group.

        Returns:
        --------
        str
            A string representation of the SL group in the format 'SL_n'.
        """
        return f"SL_{self.n}"

    def _hashable_content(self) -> tuple:
        """
        Returns a tuple containing the dimension n, used for hashing.

        Returns:
        --------
        tuple
            A tuple containing the dimension n.
        """
        return (self.n,)
