from __future__ import annotations
from developing.utils import *
from math import comb
from time import time
from functools import reduce
from itertools import combinations

superscripts = str.maketrans("23456789", "²³⁴⁵⁶⁷⁸⁹", "1")


class Poly:
    def __init__(self, data: dict[tuple[int], int], vars: list[str]) -> None:
        self.data = data
        self.vars = vars

    def _make(self, data: dict) -> Poly:
        return Poly(data, self.vars)

    def __add__(self, other: Poly | int) -> Poly:
        result = self.data.copy()
        if isinstance(other, int):
            constant = (0,)*len(self.vars)
            result[constant] = result.get(constant, 0) + other
            if result[constant] == 0:
                del result[constant]
            return self._make(result)
        for tup, coef in other.data.items():
            result[tup] = result.get(tup, 0) + coef
            if result[tup] == 0:
                del result[tup]
        return self._make(result)

    def __sub__(self, other: Poly) -> Poly:
        return self.__add__(-other)
    
    def __neg__(self):
        result = {v: -coef for v, coef in self.data.items()}
        return self._make(result)

    def __mul__(self, other: int | Poly) -> Poly:
        if isinstance(other, int):
            if other == 0:
                return self._make({})
            return self._make({k: v * other for k, v in self.data.items() if v * other != 0})
        result = {}
        for t1, c1 in self.data.items():
            for t2, c2 in other.data.items():
                key = tuple(a + b for a, b in zip(t1, t2))
                result[key] = result.get(key, 0) + c1 * c2
                if result[key] == 0:
                    del result[key]
        return self._make(result)
    
    def lift(self, lift_factor) -> Poly:
        new_data = {(v[0]-lift_factor,) + v[1:]: coef for v, coef in self.data.items()}
        return self._make(new_data)

    def __repr__(self) -> str:
        expr = " + ".join(
            "".join(
                ([str(self.data[v])] if self.data[v] != 1 or sum(v) == 0 else [])
                + [f"{self.vars[i]}{str(v[i]).translate(superscripts)}" for i in range(len(v)) if v[i] > 0]
            )
            for v in self.data.keys()
        )
        return expr

    @classmethod
    def from_sym(cls, expr: sp.Expr) -> Poly:
        lambdas = sorted(expr.free_symbols, key=lambda x: (len(x.name), x.name))
        poly = sp.Poly(expr, lambdas)
        data = {v: coef for v, coef in poly.terms()}
        return cls(data, [s.name for s in lambdas])

    @classmethod
    def from_chow(cls, expr: sp.Expr, motive: sp.Expr) -> Poly:
        lambdas = sorted(motive.free_symbols, key=lambda x: (len(x.name), x.name))
        poly = sp.Poly(expr, lambdas)
        data = {v: coef for v, coef in poly.terms()}
        return cls(data, [s.name for s in lambdas])


class HodgePoly(Poly):

    def __init__(self, data: dict, vars: list[str]) -> None:
        if len(vars) != 2:
            raise ValueError("Hodge polynomial must be of two variables")
        super().__init__(data, vars)
        self.deg = max(sum(tup) for tup in data) // 2 if self.data else 0

    def _make(self, data: dict) -> "HodgePoly":
        """Returns a HodgePoly; deg is recomputed from data automatically."""
        return HodgePoly(data, self.vars)
    
    def draw_diamond(self, val_size: int = 6, indx_len: int = 2) -> None:
        deg = self.deg
        for d in range(deg + 1):
            print((indx_len - len(str(d))) * " " + str(d) + "     ", end="")
            print(" " * int(val_size * (deg - d) / 2), end="")
            for i in range(d + 1):
                j = d - i
                coef_str = str(self.data.get((i, j), 0))
                print(coef_str + " " * (val_size - len(coef_str)), end="")
            print("")
        for d in range(deg + 1, 2 * deg + 1):
            print((indx_len - len(str(d))) * " " + str(d) + "     ", end="")
            print(" " * int(val_size * (d - deg) / 2), end="")
            for i in range(deg + 1):
                j = d - i
                if j <= deg:
                    coef_str = str(self.data.get((i, j), 0))
                    print(coef_str + " " * (val_size - len(coef_str)), end="")
            print("")
        print("")

    def is_symetric(self) -> bool:
        # check both horizontal and vertical symmetry
        deg = self.deg
        for i, j in self.data.keys():
            offset = deg - (i + j)
            if self.data[(i, j)] != self.data.get((j, i), 0) or self.data[(i, j)] != self.data.get((i + offset, j + offset), 0):
                return False
        return True

    def get_diamond_size(self) -> tuple[int, int]:
        """
        find possible sizes of smallest diamond (returns half the size, half the chopped size),
        so the actual diamond may have any size between d_size and d_size+chopped_size.
        we are assuming here that the smallest diamond will not lay inside some bigger,
        unchopped diamond starting at the beginning.
        """
        deg = self.deg
        d_size = 0
        for d in range(2*deg + 1): # antes deg+1
            if self.data.get((d, 0), 0) > 0:
                d_size += 1
            else:
                break
        chopped_size = 0
        for d in range(2*deg + 1): # antes deg+1
            if self.data.get((d_size - 1 + d, d), 0) > 0 and self.data.get((d_size - 1 + d, d - 1), 0) == 0:
                chopped_size += 1
            else:
                break
        chopped_size = ((chopped_size - 1) * 2 - 1) // 2 + 1
        return (d_size, chopped_size)

    def get_lift_factor(self) -> int:
        return max(d for d in range(0, self.deg+1) if all(x-d>=0 and y-d>=0 for (x,y) in self.data.keys()))

    def lift(self) -> tuple[int, HodgePoly]:
        d = self.get_lift_factor()
        data = {(x-d,y-d): coef for (x,y), coef in self.data.items()}
        return d, self._make(data)

    @classmethod
    def from_sym(cls, expr: sp.Expr) -> HodgePoly:
        base = super().from_sym(expr)  # returns a Poly
        return cls(base.data, base.vars)  # deg computed automatically

    @classmethod
    def from_chow(cls, motive: LambdaRingExpr | int, curve: Curve) -> HodgePoly:
        # computes hodge polynomial of motive
        X, Y = sp.Symbol("X"), sp.Symbol("Y")
        if type(motive) in {int, sp.core.numbers.One}:
            return cls({(0, 0): motive}, ["X", "Y"])
        subs = {
            L: X * Y,
            **{sym_lambda_chow_in_chow(curve, i): cls.lambda_chow2hodge(curve.g, i) for i in range(1, curve.g + 1)}
        }
        return cls.from_sym(motive.subs(subs).expand())

    @classmethod
    def lambda_chow2hodge(cls, g: int, k: int) -> sp.Expr:
        # computes hodge polynomial of lambda^k(h1(X)), where g is the genus of X
        return sum(comb(g, i) * comb(g, k - i) * sp.Symbol("X") ** i * sp.Symbol("Y") ** (k - i) for i in range(k + 1))

class HodgeFinder:
    
    def __init__(self, r: int, g: int, d: int=1) -> None:
        self.r = r
        self.g = g
        self.d = d
        self._partition_size = r-1
        self.curve = Curve("X", g)
        chow = symbolize_chow(get_motive_chow(self.curve, r, d), self.curve)
        self.m_chow = Poly.from_chow(chow, chow)
        self.m_hodge = HodgePoly.from_chow(chow, self.curve)
        self.sym2chow = {sym_lambda(self.curve, k): Poly.from_chow(sym_lambda_in_chow(self.curve, k), chow) for k in range((r**2-1)*(g-1)+1)}
        self.sym2hodge = {sym_lambda(self.curve, k): HodgePoly.from_chow(sym_lambda_in_chow(self.curve, k), self.curve) for k in range((r**2-1)*(g-1)+1)}
        self.factor2hodge = {1 + (L**k if k>0 else 0): HodgePoly.from_chow(1 + (L**k if k>0 else 0), self.curve) for k in range(self.m_hodge.deg+1)}
        self.factor2chow = {1 + (L**k if k>0 else 0): Poly.from_chow(1 + (L**k if k>0 else 0), chow) for k in range(self.m_hodge.deg+1)}
        self.visited = set()
        print("finder initialized")

    def product(self, items):
        if len(items)==0:
            return 1
        return reduce(lambda a, b: a*b, items)

    def find_motive(self) -> sp.Expr:
        res = self.motive_search_original(self.m_chow, self.m_hodge)
        return res.expand() if res is not None else None

    def motive_search_original(self, chow: LambdaRingExpr, hodge: HodgePoly) -> LambdaRingExpr:
        # returns reamining part of m covering current chow
        if any(coef<0 for coef in chow.data.values()):
            return None
        if len(chow.data) == 0:
            return 0
        frozen = frozenset(chow.data.items())
        if frozen in self.visited:
            return None
        self.visited.add(frozen)
        d_size, chopped_size = hodge.get_diamond_size()
        partitions = [partition for k in range(chopped_size+1) for partition in self.get_partitions(d_size, k, self._partition_size)]
        monoms_curve = [sym_lambda_monomial(self.curve, partition) for partition in partitions]
        monoms_chow = [self.product([self.sym2chow[sym_lambda(self.curve, k)] for k in partition]) for partition in partitions]
        monoms_hodge = [self.product([self.sym2hodge[sym_lambda(self.curve, k)] for k in partition]) for partition in partitions]
        for partition, monom_chow, monom_curve, monom_hodge in zip(partitions, monoms_chow, monoms_curve, monoms_hodge):
            factor = 1 + (L**(hodge.deg - sum(partition)) if hodge.deg - sum(partition) > 0 else 0)
            factor_chow = self.factor2chow[factor]
            lift_factor, new_hodge = (hodge - self.factor2hodge[factor]*monom_hodge).lift()
            m = monom_curve*factor
            new_chow = (chow - factor_chow*monom_chow).lift(lift_factor)
            candidate = self.motive_search_original(new_chow, new_hodge)
            if candidate is not None:
                return m + (L**lift_factor)*candidate
        return None
    
    def motive_search(self, chow: LambdaRingExpr, hodge: HodgePoly):
        # returns reamining part of m covering current chow
        if any(coef<0 for coef in chow.data.values()):
            return None
        if len(chow.data) == 0:
            return 0
        d_size, chopped_size = hodge.get_diamond_size()
        partitions = [partition for k in range(chopped_size) for partition in self.get_partitions(d_size, k, length=self._partition_size)] + [partition for k in range(d_size-1, 0, -1) for partition in self.get_partitions(k, 0, self._partition_size)]   # por alguna razon chopped size + 1
        monoms_hodge = [self.product([self.sym2hodge[sym_lambda(self.curve, k)] for k in partition]) for partition in partitions]
        monoms_curve = [sym_lambda_monomial(self.curve, partition) for partition in partitions]
        monoms_chow = [self.product([self.sym2chow[sym_lambda(self.curve, k)] for k in partition]) for partition in partitions]
        sums_idxs = self.row_search(hodge, monoms_hodge)
        print(sums_idxs)
        print(monoms_curve)
        hodge.draw_diamond()
        for sum_idxs in sums_idxs:
            sum_chow = 0
            sum_hodge = 0
            sum_m = 0
            for idx in sum_idxs:
                partition = partitions[idx]
                monom_curve = monoms_curve[idx]
                monom_chow = monoms_chow[idx]
                monom_hodge = monoms_hodge[idx]
                factor = 1 + (L**(hodge.deg - sum(partition)) if hodge.deg - sum(partition) > 0 else 0)
                sum_hodge = self.factor2hodge[factor]*monom_hodge + sum_hodge
                sum_chow = self.factor2chow[factor]*monom_chow + sum_chow
                sum_m += factor*monom_curve
            lift_factor, new_hodge = (hodge-sum_hodge).lift()
            new_chow = (chow-sum_chow).lift(lift_factor)
            candidate = self.motive_search(new_chow, new_hodge)
            if candidate is not None:
                return sum_m + (L**lift_factor)*candidate
        return None

    
    def row_search(self, hodge: HodgePoly, hodge_monoms: list[HodgePoly]) -> list[tuple[int, ...]]:
        results = []
        n = len(hodge_monoms)

        def backtrack(start: int, current_indices: list[int], current_sum: HodgePoly):
            if self.coincide(hodge, current_sum):
                if n == 1 or tuple(current_indices) != (n - 1,):
                    results.append(tuple(current_indices))
                return

            for i in range(start, n):
                new_sum = current_sum + hodge_monoms[i]
                if all(coef > 0 for coef in (hodge - new_sum).data.values()):
                    backtrack(i + 1, current_indices + [i], new_sum)

        backtrack(0, [], HodgePoly({}, ["X", "Y"]))
        return results
        
    def coincide(self, hodge1: HodgePoly, hodge2: HodgePoly):
        return all(hodge1.data.get((d, 0),0) == hodge2.data.get((d, 0),0) for d in range(min(hodge1.deg, hodge2.deg)+1))
        
        
    def get_partitions(self, d_size: int, chopped_size: int, length: int) -> list:
        """
        get possible generators of diamond size
        -> non-decreasing tuples of length <= `length` summing to (d_size + chopped_size - 1)
        such that sum of overparts = chopped_size
        """
        target_sum = d_size + chopped_size - 1
        results = []

        def backtrack(remaining_sum, remaining_chop, current, min_val):
            # Base case: if no more elements to add
            if remaining_sum == 0:
                if remaining_chop == 0:
                    results.append(tuple(current))
                return

            for val in range(min_val, remaining_sum + 1):
                excess = max(0, val - self.g)
                if excess > remaining_chop:
                    break  # val will only grow, no point continuing
                backtrack(
                    remaining_sum - val,
                    remaining_chop - excess,
                    current + [val],
                    val  # non-decreasing: next val >= current
                )

        backtrack(target_sum, chopped_size, [], 1)
        return [v for v in results if len(v) <= length]
        

if __name__ == "__main__":
    """
    IDEA
    Buscamos diamante
    - si diamante no rompe, sacamos lambda, pero hay que ver si es producto etc
    - si diamante rompe, probablemente sea por ks mayores a g, y la suma de los sobrepasos por encima de g del monomio son el numero de filas con ceros (simétrico)
    """
    r = 2
    g = 2
    finder = HodgeFinder(r, g)
    finder.m_hodge.draw_diamond()


    curve = finder.curve
    k1, k2, k3 = 2, 2, 2
    m = sym_lambda_in_chow(curve, k1) * sym_lambda_in_chow(curve, k2) * sym_lambda_in_chow(curve, k3)
    hodge = HodgePoly.from_chow(m, curve)
    hodge.draw_diamond()