from .general_groups import A


class PSL(A):
    """
    TODO docs
    """

    def __new__(cls, n: int, *args, **kwargs):
        """
        Creates a new instance of the PSL class with the specified dimension n.

        Parameters:
        -----------
        n : int
            The dimension of the symplectic group PSL(n).
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns:
        --------
        PSL
            A new instance of the PSL class.
        """
        new_sl = A.__new__(cls, n - 1)
        new_sl.n = n
        return new_sl

    def __repr__(self) -> str:
        """
        Returns a string representation of the PSL group.

        Returns:
        --------
        str
            A string representation of the PSL group in the format 'PSL_n'.
        """
        return f"PSL_{self.n}"

    def _hashable_content(self) -> tuple:
        """
        Returns a tuple containing the dimension n, used for hashing.

        Returns:
        --------
        tuple
            A tuple containing the dimension n.
        """
        return (self.n,)