from .semisimple_g import SemisimpleG
from itertools import chain


"""
Set of classes describing the motives of complex algebraic Lie groups
in terms of their classification in types An, Bn, Cn, Dn, as well as
classes for the exceptional groups E6, E7, E8, F4, and G2.
"""


class A(SemisimpleG):
    """
    Represents the Grothendieck motive of a semisimple complex algebraic group of type A_n.

    Attributes:
    -----------
    n : int
        The rank of the group A(n).

    Methods:
    --------
    __new__(cls, n: int, *args, **kwargs) -> 'A':
        Creates a new instance of the A group with the specified rank n.
    __repr__(self) -> str:
        Returns a string representation of the A group.
    _hashable_content(self) -> tuple:
        Returns a tuple containing the rank n, used for hashing.
    """

    def __new__(cls, n: int, *args, **kwargs) -> "A":
        """
        Creates a new instance of the A group with the specified rank n.

        Parameters:
        -----------
        n : int
            The rank of the group A(n).
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns:
        --------
        A
            A new instance of the A class.
        """
        new_sl = SemisimpleG.__new__(cls, list(range(2, n + 2)), n * (n + 1))
        return new_sl

    def __init__(self, n: int, *args, **kwargs) -> None:
        """
        Initializes the A class with the specified rank n.

        Parameters:
        -----------
        n : int
            The rank of the group A(n).
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(list(range(2, n + 2)), n * (n + 1))
        self.n: int = n


class B(SemisimpleG):
    """
    Represents the Grothendieck motive of a semisimple complex algebraic group of type B_n.

    Attributes:
    -----------
    n : int
        The rank of the group B(n).

    Methods:
    --------
    __new__(cls, n: int, *args, **kwargs) -> 'B':
        Creates a new instance of the B group with the specified rank n.
    __repr__(self) -> str:
        Returns a string representation of the B group.
    _hashable_content(self) -> tuple:
        Returns a tuple containing the rank n, used for hashing.
    """

    def __new__(cls, n: int, *args, **kwargs) -> "B":
        """
        Creates a new instance of the B group with the specified rank n.

        Parameters:
        -----------
        n : int
            The rank of the group B(n).
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns:
        --------
        B
            A new instance of the B class.
        """
        new_sl = SemisimpleG.__new__(cls, list(range(2, 2 * n + 1, 2)), n * (2 * n + 1))
        new_sl.n = n
        return new_sl

    def __init__(self, n: int, *args, **kwargs) -> None:
        """
        Initializes the B class with the specified rank n.

        Parameters:
        -----------
        n : int
            The rank of the group B(n).
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(list(range(2, 2 * n + 1, 2)), n * (2 * n + 1))
        self.n: int = n


class C(SemisimpleG):
    """
    Represents the Grothendieck motive of a semisimple complex algebraic group of type C_n.

    Attributes:
    -----------
    n : int
        The rank of the group C(n).

    Methods:
    --------
    __new__(cls, n: int, *args, **kwargs) -> 'C':
        Creates a new instance of the C group with the specified rank n.
    __repr__(self) -> str:
        Returns a string representation of the C group.
    _hashable_content(self) -> tuple:
        Returns a tuple containing the rank n, used for hashing.
    """

    def __new__(cls, n: int, *args, **kwargs) -> "C":
        """
        Creates a new instance of the C group with the specified rank n.

        Parameters:
        -----------
        n : int
            The rank of the group C(n).
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns:
        --------
        C
            A new instance of the C class.
        """
        new_sl = SemisimpleG.__new__(cls, list(range(2, 2 * n + 1, 2)), n * (2 * n + 1))
        new_sl.n = n
        return new_sl

    def __init__(self, n: int, *args, **kwargs) -> None:
        """
        Initializes the C class with the specified rank n.

        Parameters:
        -----------
        n : int
            The rank of the group C(n).
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(list(range(2, 2 * n + 1, 2)), n * (2 * n + 1))
        self.n: int = n


class D(SemisimpleG):
    """
    Represents the Grothendieck motive of a semisimple complex algebraic group of type D_n.

    Attributes:
    -----------
    n : int
        The rank of the group D(n).

    Methods:
    --------
    __new__(cls, n: int, *args, **kwargs) -> 'D':
        Creates a new instance of the D group with the specified rank n.
    __repr__(self) -> str:
        Returns a string representation of the D group.
    _hashable_content(self) -> tuple:
        Returns a tuple containing the rank n, used for hashing.
    """

    def __new__(cls, n: int, *args, **kwargs) -> "D":
        """
        Creates a new instance of the D group with the specified rank n.

        Parameters:
        -----------
        n : int
            The rank of the group D(n).
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns:
        --------
        D
            A new instance of the D class.
        """
        new_sl = SemisimpleG.__new__(
            cls, list(range(2, 2 * n, 2)) + [n], n * (2 * n - 1)
        )
        new_sl.n = n
        return new_sl

    def __init__(self, n: int, *args, **kwargs) -> None:
        """
        Initializes the D class with the specified rank n.

        Parameters:
        -----------
        n : int
            The rank of the group D(n).
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(list(range(2, 2 * n - 1, 2)) + [n], n * (2 * n - 1))
        self.n: int = n


class E(SemisimpleG):
    """
    Represents the Grothendieck motive of an exceptional group of type E_n.

    Attributes:
    -----------
    n : int
        The rank of the group E(n).

    Methods:
    --------
    __new__(cls, n: int, *args, **kwargs) -> 'E':
        Creates a new instance of the E group with the specified rank n.
    __repr__(self) -> str:
        Returns a string representation of the E group.
    _hashable_content(self) -> tuple:
        Returns a tuple containing the rank n, used for hashing.
    """

    def __new__(cls, n: int, *args, **kwargs) -> "E":
        """
        Creates a new instance of the E group with the specified rank n.

        Parameters:
        -----------
        n : int
            The rank of the group E(n).
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
            new_sl = SemisimpleG.__new__(cls, [2, 5, 6, 8, 9, 12], 78)
        elif n == 7:
            new_sl = SemisimpleG.__new__(cls, [2, 6, 8, 10, 12, 14, 18], 133)
        elif n == 8:
            new_sl = SemisimpleG.__new__(cls, [2, 8, 12, 14, 18, 20, 24, 30], 248)
        else:
            raise ValueError("The rank of the group E(n) must be 6, 7, or 8.")

        new_sl.n = n
        return new_sl

    def __init__(self, n: int, *args, **kwargs) -> None:
        """
        Initializes the E class with the specified rank n.

        Parameters:
        -----------
        n : int
            The rank of the group E(n).
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
        self.n: int = n


class F4(SemisimpleG):
    """
    Represents the Grothendieck motive of the exceptional group F4.

    Attributes:
    -----------
    n : int
        The rank of the group F4. It is always 4.

    Methods:
    --------
    __new__(cls, *args, **kwargs) -> 'F4':
        Creates a new instance of the F4 group.
    __repr__(self) -> str:
        Returns a string representation of the F4 group.
    _hashable_content(self) -> tuple:
        Returns a tuple containing the rank n, used for hashing.
    """

    def __new__(cls, *args, **kwargs) -> "F4":
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
        new_sl = SemisimpleG.__new__(cls, [2, 6, 8, 12], 52)
        new_sl.n = 4
        return new_sl

    def __init__(self, *args, **kwargs) -> None:
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
        self.n: int = 4


class G2(SemisimpleG):
    """
    Represents the Grothendieck motive of the exceptional group G2.

    Attributes:
    -----------
    n : int
        The rank of the group G2. It is always 2.

    Methods:
    --------
    __new__(cls, *args, **kwargs) -> 'G2':
        Creates a new instance of the G2 group.
    __repr__(self) -> str:
        Returns a string representation of the G2 group.
    _hashable_content(self) -> tuple:
        Returns a tuple containing the rank n, used for hashing.
    """

    def __new__(cls, *args, **kwargs) -> "G2":
        """
        Creates a new instance of the G2 group.

        Parameters:
        -----------
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns:
        --------
        G2
            A new instance of the G2 class.
        """
        new_sl = SemisimpleG.__new__(cls, [2, 6], 28)
        new_sl.n = 2
        return new_sl

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the G2 class.

        Parameters:
        -----------
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__([2, 6], 28)
        self.n: int = 2
