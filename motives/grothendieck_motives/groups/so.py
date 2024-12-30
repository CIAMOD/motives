from .general_groups import B, D


# TODO docs


class SOBase:
    """
    Base class for special orthogonal groups.
    """

    def __repr__(self) -> str:
        return f"SO_{self.n}"

    def _hashable_content(self) -> tuple:
        return (self.n,)


class _SOOdd(SOBase, B):
    """
    Represents the special orthogonal group SO(n) for odd n.
    """

    def __new__(cls, n: int, *args, **kwargs):
        instance = B.__new__(cls, (n - 1) // 2)
        instance.n = n
        return instance

    def __init__(self, n: int, *args, **kwargs) -> None:
        B.__init__(self, (n - 1) // 2)
        self.n = n


class _SOEven(SOBase, D):
    """
    Represents the special orthogonal group SO(n) for even n.
    """

    def __new__(cls, n: int, *args, **kwargs):
        instance = D.__new__(cls, n // 2)
        instance.n = n
        return instance

    def __init__(self, n: int, *args, **kwargs) -> None:
        D.__init__(self, n // 2)
        self.n = n


def SO(n: int, *args, **kwargs):
    if n % 2 == 1:
        return _SOOdd(n, *args, **kwargs)
    else:
        return _SOEven(n, *args, **kwargs)
