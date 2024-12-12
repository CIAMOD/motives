from g import G


class SP(G):
    n: int

    def __new__(cls, n: int, *args, **kwargs):
        new_sl = G.__new__(cls, range(2, 2 * n + 1, 2), n * (2 * n + 1))
        new_sl.n = n
        return new_sl

    def __repr__(self) -> str:
        return f"SP_{self.n}"

    def _hashable_content(self) -> tuple:
        return (self.n,)
