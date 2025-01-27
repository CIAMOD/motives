from .general_groups import A


class SL(A):
    """
    Represents the special linear group SL(n) as a Grothendieck motive.
    This class inherits from the A class and represents the special linear group SL(n),
    which consists of n x n matrices with determinant 1. The motive is constructed
    using the range of integers from 2 to n (inclusive) and the dimension n² - 1.

    Attributes
    ----------
    n : int
        The dimension of the special linear group SL(n).

    Methods
    -------
    __new__(cls, n: int, *args: Any, **kwargs: Any) -> "SL":
        Creates a new instance of the SL class with the specified dimension n.
    __repr__(self) -> str:
        Returns a string representation of the SL group.
    _hashable_content(self) -> tuple:
        Returns a tuple containing the dimension n, used for hashing.
    """

    def __new__(cls, n: int, *args, **kwargs) -> "SL":
        """
        Creates a new instance of the SL class with the specified dimension n.

        Parameters
        ----------
        n : int
            The dimension of the special linear group SL(n).
        *args : Any
            Additional positional arguments.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        SL
            A new instance of the SL class.
        """
        new_sl = A.__new__(cls, n - 1)
        new_sl.n = n
        return new_sl

    def __repr__(self) -> str:
        """
        Returns a string representation of the SL group.

        Returns
        -------
        str
            A string representation of the SL group in the format 'SL_n'.
        """
        return f"SL_{self.n}"

    def _hashable_content(self) -> tuple:
        """
        Returns a tuple containing the dimension n, used for hashing.

        Returns
        -------
        tuple
            A tuple containing the dimension n.
        """
        return (self.n,)
