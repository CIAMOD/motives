from g import G


class SP(G):
    """
    Represents the symplectic group SP(n) as a Grothendieck motive.
    This class inherits from the G class and represents the symplectic group SP(n),
    which consists of 2n x 2n matrices preserving a symplectic form. The motive is constructed
    using a combination of even integers from 2 to 2n and the integer n, with the dimension n * (2n + 1).
    Attributes:
    -----------
    n : int
        The dimension of the symplectic group SP(n).
    Methods:
    --------
    __new__(cls, n: int, *args, **kwargs)
        Creates a new instance of the SP class with the specified dimension n.
    __repr__() -> str
        Returns a string representation of the SP group.
    _hashable_content() -> tuple
        Returns a tuple containing the dimension n, used for hashing.
    """

    def __new__(cls, n: int, *args, **kwargs):
        """
        Creates a new instance of the SP class with the specified dimension n.

        Parameters:
        -----------
        n : int
            The dimension of the symplectic group SP(n).
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns:
        --------
        SP
            A new instance of the SP class.
        """
        new_sl = G.__new__(cls, range(2, 2 * n + 1, 2), n * (2 * n + 1))
        new_sl.n = n
        return new_sl

    def __repr__(self) -> str:
        """
        Returns a string representation of the SP group.

        Returns:
        --------
        str
            A string representation of the SP group in the format 'SP_n'.
        """
        return f"SP_{self.n}"

    def _hashable_content(self) -> tuple:
        """
        Returns a tuple containing the dimension n, used for hashing.

        Returns:
        --------
        tuple
            A tuple containing the dimension n.
        """
        return (self.n,)
