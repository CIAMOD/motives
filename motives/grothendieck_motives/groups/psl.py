from .general_groups import A


class PSL(A):
    """
    Represents the Projective Special Linear group PSL(n).

    The PSL(n) group is defined as the quotient of the Special Linear group SL(n) by its center.
    """

    def __new__(cls, n: int) -> "PSL":
        """
        Creates a new instance of the PSL class with the specified dimension n.

        Parameters
        ----------
        n : int
            The dimension of the Projective Special Linear group PSL(n).

        Returns
        -------
        PSL
            A new instance of the PSL class.
        """
        new_sl = A.__new__(cls, n - 1)
        new_sl.n = n
        return new_sl

    def __repr__(self) -> str:
        """
        Returns a string representation of the PSL group.

        Returns
        -------
        str
            A string representation of the PSL group in the format 'PSL_n'.
        """
        return f"PSL_{self.n}"

    def _hashable_content(self) -> tuple:
        """
        Returns a tuple containing the dimension n, used for hashing.

        Returns
        -------
        tuple
            A tuple containing the dimension n.
        """
        return (self.n,)
