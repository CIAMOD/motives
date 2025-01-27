from .general_groups import B, D


class SOBase:
    """
    Base class for special orthogonal groups (SO(n)).

    Attributes
    ----------
    n : int
        The dimension of the special orthogonal group.
    """

    n: int  # Dimension of the SO group

    def __repr__(self) -> str:
        """Return the string representation of the SO group."""
        return f"SO_{self.n}"

    def _hashable_content(self) -> tuple:
        """
        Provide a hashable content representation of the SO group.

        Returns
        -------
        tuple
            A tuple containing the dimension 'n'.
        """
        return (self.n,)


class _SOOdd(SOBase, B):
    """
    Represents the special orthogonal group SO(n) for odd n.

    Inherits from SOBase and general group class B.

    Parameters
    ----------
    n : int
        The dimension of the special orthogonal group (must be odd).
    *args : Any
        Additional positional arguments.
    **kwargs : Any
        Additional keyword arguments.
    """

    def __new__(cls, n: int, *args, **kwargs):
        """
        Create a new instance of the _SOOdd class with the specified odd dimension.

        Parameters
        ----------
        n : int
            The dimension of SO(n) (must be odd).
        *args : Any
            Additional positional arguments.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        _SOOdd
            A new instance of the _SOOdd class.
        """
        instance = B.__new__(cls, (n - 1) // 2)
        instance.n = n
        return instance

    def __init__(self, n: int, *args, **kwargs) -> None:
        """
        Initialize the _SOOdd instance.

        Parameters
        ----------
        n : int
            The dimension of SO(n) (must be odd).
        *args : Any
            Additional positional arguments.
        **kwargs : Any
            Additional keyword arguments.
        """
        B.__init__(self, (n - 1) // 2)
        self.n = n


class _SOEven(SOBase, D):
    """
    Represents the special orthogonal group SO(n) for even n.

    Inherits from SOBase and general group class D.

    Parameters
    ----------
    n : int
        The dimension of the special orthogonal group (must be even).
    *args : Any
        Additional positional arguments.
    **kwargs : Any
        Additional keyword arguments.
    """

    def __new__(cls, n: int, *args, **kwargs):
        """
        Create a new instance of the _SOEven class with the specified even dimension.

        Parameters
        ----------
        n : int
            The dimension of SO(n) (must be even).
        *args : Any
            Additional positional arguments.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        _SOEven
            A new instance of the _SOEven class.
        """
        instance = D.__new__(cls, n // 2)
        instance.n = n
        return instance

    def __init__(self, n: int, *args, **kwargs) -> None:
        """
        Initialize the _SOEven instance.

        Parameters
        ----------
        n : int
            The dimension of SO(n) (must be even).
        *args : Any
            Additional positional arguments.
        **kwargs : Any
            Additional keyword arguments.
        """
        D.__init__(self, n // 2)
        self.n = n


def SO(n: int, *args, **kwargs) -> SOBase:
    """
    Factory function to create an instance of the special orthogonal group SO(n).

    Parameters
    ----------
    n : int
        The dimension of the special orthogonal group.
    *args : Any
        Additional positional arguments.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    _SOOdd or _SOEven
        An instance of _SOOdd if n is odd, or _SOEven if n is even.
    """
    if n % 2 == 1:
        return _SOOdd(n, *args, **kwargs)
    else:
        return _SOEven(n, *args, **kwargs)
