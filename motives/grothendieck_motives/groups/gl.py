from .semisimple_g import SemisimpleG


class GL(SemisimpleG):
    """
    Represents the Grothendieck motive of the general linear group GL(n, C).

    The motive of GL(n, C) is represented by the product
        [GL(n, C)] = ∏_{k=0}^{n-1} (L^n - L^k),
    where L is the Lefschetz motive. This class supports operations related to Adams
    and Lambda transformations.

    Attributes:
    -----------
    n : int
        The dimension of the vector space.
    """

    def __new__(cls, n: int, *args: tuple, **kwargs: dict) -> "GL":
        """
        Creates a new instance of the GL group with the specified rank n.

        Parameters:
        -----------
        n : int
            The rank of the group GL(n).
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns:
        --------
        GL
            A new instance of the GL class.
        """
        new_gl = SemisimpleG.__new__(cls, list(range(2, n + 2)), n**2)
        return new_gl

    def __init__(self, n: int, *args: tuple, **kwargs: dict) -> None:
        """
        Initializes a GL(n, C) instance with the specified rank n as
        a type A_n group.

        Parameters:
        -----------
        n : int
            The rank of the group.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(list(range(2, n + 2)), n**2)
        self.n: int = n

    def __repr__(self) -> str:
        """
        Returns the string representation of the group GL(n, C).

        Returns:
        --------
        str
            A string representation in the form "GL_n".
        """
        return f"GL_{self.n}"

    def _hashable_content(self) -> tuple:
        """
        Returns the hashable content of the group GL_n.

        Returns:
        --------
        tuple
            A tuple containing the rank of the group GL_n.
        """
        return (self.n,)
