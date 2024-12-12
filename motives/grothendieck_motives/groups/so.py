from g import G
from itertools import chain


class SO(G):
    n: int

    def __new__(cls, n: int, *args, **kwargs):
        new_sl = G.__new__(cls, chain(range(2, 2 * n - 1, 2), (n,)), n * (2 * n - 1))
        new_sl.n = n
        return new_sl

    def __repr__(self) -> str:
        return f"SO_{self.n}"

    def _hashable_content(self) -> tuple:
        return (self.n,)
