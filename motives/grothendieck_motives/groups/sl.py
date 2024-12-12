from g import G


class SL(G):
    n: int

    def __new__(cls, n: int, *args, **kwargs):
        new_sl = G.__new__(cls, range(2, n + 1), n**2 - 1)
        new_sl.n = n
        return new_sl

    def __repr__(self) -> str:
        return f"SL_{self.n}"

    def _hashable_content(self) -> tuple:
        return (self.n,)
