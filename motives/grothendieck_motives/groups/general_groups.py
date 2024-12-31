from .g import G
from itertools import chain


# TODO revisar docs


class A(G):
    """
    Represents the general group A(n) as a Grothendieck motive.

    Attributes:
    -----------
    n : int
        The dimension of the group A(n).

    Methods:
    --------
    __new__(cls, n: int, *args, **kwargs)
        Creates a new instance of the A group with the specified dimension n.
    __repr__() -> str
        Returns a string representation of the A group.
    _hashable_content() -> tuple
        Returns a tuple containing the dimension n, used for hashing.
    """

    def __new__(cls, n: int, *args, **kwargs):
        """
        Creates a new instance of the A group with the specified dimension n.

        Parameters:
        -----------
        n : int
            The dimension of the group A(n).
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns:
        --------
        A
            A new instance of the A class.
        """
        new_sl = G.__new__(cls, list(range(2, n + 2)), n * (n + 1))
        return new_sl

    def __init__(self, n: int, *args, **kwargs):
        """
        Initializes the A class with the specified dimension n.

        Parameters:
        -----------
        n : int
            The dimension of the group A(n).
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(list(range(2, n + 2)), n * (n + 1))
        self.n = n


class B(G):
    """
    Represents the general group B(n) as a Grothendieck motive.

    Attributes:
    -----------
    n : int
        The dimension of the group B(n).

    Methods:
    --------
    __new__(cls, n: int, *args, **kwargs)
        Creates a new instance of the B group with the specified dimension n.
    __repr__() -> str
        Returns a string representation of the B group.
    _hashable_content() -> tuple
        Returns a tuple containing the dimension n, used for hashing.
    """

    def __new__(cls, n: int, *args, **kwargs):
        """
        Creates a new instance of the B group with the specified dimension n.

        Parameters:
        -----------
        n : int
            The dimension of the group B(n).
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns:
        --------
        B
            A new instance of the B class.
        """
        new_sl = G.__new__(cls, list(range(2, 2 * n + 1, 2)), n * (2 * n + 1))
        new_sl.n = n
        return new_sl

    def __init__(self, n: int, *args, **kwargs):
        """
        Initializes the B class with the specified dimension n.

        Parameters:
        -----------
        n : int
            The dimension of the group B(n).
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(list(range(2, 2 * n + 1, 2)), n * (2 * n + 1))
        self.n = n


class C(G):
    """
    Represents the general group C(n) as a Grothendieck motive.

    Attributes:
    -----------
    n : int
        The dimension of the group C(n).

    Methods:
    --------
    __new__(cls, n: int, *args, **kwargs)
        Creates a new instance of the C group with the specified dimension n.
    __repr__() -> str
        Returns a string representation of the C group.
    _hashable_content() -> tuple
        Returns a tuple containing the dimension n, used for hashing.
    """

    def __new__(cls, n: int, *args, **kwargs):
        """
        Creates a new instance of the C group with the specified dimension n.

        Parameters:
        -----------
        n : int
            The dimension of the group C(n).
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns:
        --------
        C
            A new instance of the C class.
        """
        new_sl = G.__new__(cls, list(range(2, 2 * n + 1, 2)), n * (2 * n + 1))
        new_sl.n = n
        return new_sl

    def __init__(self, n: int, *args, **kwargs):
        """
        Initializes the C class with the specified dimension n.

        Parameters:
        -----------
        n : int
            The dimension of the group C(n).
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(list(range(2, 2 * n + 1, 2)), n * (2 * n + 1))
        self.n = n


class D(G):
    """
    Represents the general group D(n) as a Grothendieck motive.

    Attributes:
    -----------
    n : int
        The dimension of the group D(n).

    Methods:
    --------
    __new__(cls, n: int, *args, **kwargs)
        Creates a new instance of the D group with the specified dimension n.
    __repr__() -> str
        Returns a string representation of the D group.
    _hashable_content() -> tuple
        Returns a tuple containing the dimension n, used for hashing.
    """

    def __new__(cls, n: int, *args, **kwargs):
        """
        Creates a new instance of the D group with the specified dimension n.

        Parameters:
        -----------
        n : int
            The dimension of the group D(n).
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns:
        --------
        D
            A new instance of the D class.
        """
        new_sl = G.__new__(cls, list(range(2, 2 * n - 1, 2)) + [n], n * (2 * n - 1))
        new_sl.n = n
        return new_sl

    def __init__(self, n: int, *args, **kwargs):
        """
        Initializes the D class with the specified dimension n.

        Parameters:
        -----------
        n : int
            The dimension of the group D(n).
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(list(range(2, 2 * n - 1, 2)) + [n], n * (2 * n - 1))
        self.n = n


class E(G):
    """
    Represents the general group E(n) as a Grothendieck motive.

    Attributes:
    -----------
    n : int
        The dimension of the group E(n).

    Methods:
    --------
    __new__(cls, n: int, *args, **kwargs)
        Creates a new instance of the E group with the specified dimension n.
    __repr__() -> str
        Returns a string representation of the E group.
    _hashable_content() -> tuple
        Returns a tuple containing the dimension n, used for hashing.
    """

    def __new__(cls, n: int, *args, **kwargs):
        """
        Creates a new instance of the E group with the specified dimension n.

        Parameters:
        -----------
        n : int
            The dimension of the group E(n).
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns:
        --------
        E
            A new instance of the E class.
        """
        if n == 6:
            new_sl = G.__new__(cls, [2, 5, 6, 8, 9, 12], 78)
        elif n == 7:
            new_sl = G.__new__(cls, [2, 6, 8, 10, 12, 14, 18], 133)
        elif n == 8:
            new_sl = G.__new__(cls, [2, 8, 12, 14, 18, 20, 24, 30], 248)
        else:
            raise ValueError("The dimension of the group E(n) must be 6, 7, or 8.")

        new_sl.n = n
        return new_sl

    def __init__(self, n: int, *args, **kwargs):
        """
        Initializes the E class with the specified dimension n.

        Parameters:
        -----------
        n : int
            The dimension of the group E(n).
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        if n == 6:
            super().__init__([2, 5, 6, 8, 9, 12], 78)
        elif n == 7:
            super().__init__([2, 6, 8, 10, 12, 14, 18], 133)
        elif n == 8:
            super().__init__([2, 8, 12, 14, 18, 20, 24, 30], 248)
        self.n = n


class F4(G):
    """
    Represents the general group F4 as a Grothendieck motive.

    Attributes:
    -----------
    n : int
        The dimension of the group F4. It is always 4.

    Methods:
    --------
    __new__(cls, n: int, *args, **kwargs)
        Creates a new instance of the F4 group with the specified dimension n.
    __repr__() -> str
        Returns a string representation of the F4 group.
    _hashable_content() -> tuple
        Returns a tuple containing the dimension n, used for hashing.
    """

    def __new__(cls, *args, **kwargs):
        """
        Creates a new instance of the F4 group.

        Parameters:
        -----------
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns:
        --------
        F4
            A new instance of the F4 class.
        """
        new_sl = G.__new__(cls, [2, 6, 8, 12], 52)
        new_sl.n = 4
        return new_sl

    def __init__(self, *args, **kwargs):
        """
        Initializes the F4 class.

        Parameters:
        -----------
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__([2, 6, 8, 12], 52)
        self.n = 4
