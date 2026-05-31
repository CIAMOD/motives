from __future__ import annotations
from developing.utils import *
from math import comb
from copy import deepcopy

superscripts = str.maketrans("23456789", "²³⁴⁵⁶⁷⁸⁹", "1")


class Poly:
    def __init__(self, data: dict, vars: list[str]) -> None:
        self.data = data  # {(2,1,0): coef, ...}
        self.vars = vars

    def _make(self, data: dict) -> "Poly":
        """Factory used by operators — subclasses override to return their own type."""
        return Poly(data, self.vars)

    def __add__(self, other: "Poly") -> "Poly":
        result = self.data.copy()
        for tup, coef in other.data.items():
            result[tup] = result.get(tup, 0) + coef
            if result[tup] == 0:
                del result[tup]
        return self._make(result)

    def __sub__(self, other: "Poly") -> "Poly":
        result = self.data.copy()
        for tup, coef in other.data.items():
            result[tup] = result.get(tup, 0) - coef
            if result[tup] == 0:
                del result[tup]
        return self._make(result)

    def __mul__(self, other: "int | Poly") -> "Poly":
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
    def from_sym(cls, expr: "sp.Expr") -> "Poly":
        lambdas = sorted(expr.free_symbols, key=lambda x: (len(x.name), x.name))
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
        # find possible sizes of smallest diamond (returns half the size, half the chopped size),
        # so the actual diamond may have any size between d_size and d_size+chopped_size.
        # we are assuming here that the smallest diamond will not lay inside some bigger,
        # unchopped diamond starting at the beginning.
        deg = self.deg
        d_size = 0
        for d in range(deg + 1):
            if self.data.get((d, 0), 0) > 0:
                d_size += 1
            else:
                break
        chopped_size = 0
        for d in range(deg + 1):
            if self.data.get((d_size - 1 + d, d), 0) > 0 and self.data.get((d_size - 1 + d, d - 1), 0) == 0:
                chopped_size += 1
            else:
                break
        chopped_size = ((chopped_size - 1) * 2 - 1) // 2 + 1
        return (d_size, chopped_size)

    @classmethod
    def from_sym(cls, expr: sp.Expr) -> HodgePoly:
        base = super().from_sym(expr)  # returns a Poly
        return cls(base.data, base.vars)  # deg computed automatically

    @classmethod
    def from_chow(cls, motive: "LambdaRingExpr | int", curve: "Curve") -> "HodgePoly":
        # computes hodge polynomial of motive
        X, Y = sp.Symbol("X"), sp.Symbol("Y")
        if type(motive) == int:
            return cls({(0, 0): motive}, ["X", "Y"])
        subs = {
            L: X * Y,
            **{sym_lambda_chow_in_chow(curve, i): cls.lambda_chow2hodge(curve.g, i) for i in range(1, curve.g + 1)}
        }
        return cls.from_sym(motive.subs(subs).expand())

    @classmethod
    def lambda_chow2hodge(cls, g: int, k: int) -> "sp.Expr":
        # computes hodge polynomial of lambda^k(h1(X)), where g is the genus of X
        return sum(comb(g, i) * comb(g, k - i) * sp.Symbol("X") ** i * sp.Symbol("Y") ** (k - i) for i in range(k + 1))

class HodgeFinder:
    
    def __init__(self, r: int, g: int, d: int=1) -> None:
        self.r = r
        self.g = g
        self.d = d
        self.curve = Curve("X", g)
        self.m_chow = symbolize_chow(get_motive_chow(self.curve, r, d), self.curve)
        self.m_hodge = HodgePoly.from_chow(self.m_chow, self.curve)

    def find_motive(self) -> sp.Expr:
        return self.motive_search(self.m_chow, self.m_hodge).expand()

    def motive_search(self, chow: LambdaRingExpr, hodge: HodgePoly):
        # returns reamining part of m covering current chow
        if any(term.could_extract_minus_sign() for term in chow.as_ordered_terms()):
            return None
        if chow == 0:
            return 0
        d_size, chopped_size = hodge.get_diamond_size()
        partitions = [partition for k in range(chopped_size+1) for partition in self.get_partitions(d_size, k)]
        monoms_chow = [sym_lambda_in_chow_monomial(self.curve, partition) for partition in partitions]
        monoms_curve = [sym_lambda_monomial(self.curve, partition) for partition in partitions]
        for partition, monom_chow, monom_curve in zip(partitions, monoms_chow, monoms_curve):
            factor = 1 + (L**(hodge.deg - sum(partition)) if hodge.deg - sum(partition) > 0 else 0)
            lift_factor, new_hodge = self.hodge_lift(hodge - HodgePoly.from_chow((factor*monom_chow).expand(), self.curve))
            m = monom_curve*factor
            new_chow = ((chow - factor*monom_chow)/lift_factor).expand()
            candidate = self.motive_search(new_chow, new_hodge)
            if candidate is not None:
                return m + lift_factor*candidate
        return None
        
        
    def get_partitions(self, d_size: int, chopped_size: int) -> list: # (claude)
        """
        get possible generators of diamond size
        -> paritionns summing to (d_size + chopped_size - 1) such that sum of overparts = chopped_size
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
        return results

    
    def hodge_lift(self, poly: HodgePoly, deg: int=-1) -> tuple[LambdaRingExpr, HodgePoly]:
        if deg == -1:
            deg = poly.deg
        l_exp = 0
        for d in range(0, deg, 2):
            if all((i,d-i) not in poly.data.keys() for i in range(0, d+1)):
                l_exp += 1
            else:
                break
        new_data = {(i-l_exp, j-l_exp): coef for (i,j), coef in poly.data.items()}
        new_poly = HodgePoly(new_data, poly.vars)
        return L**l_exp, new_poly
        

if __name__ == "__main__":
    """
    IDEA
    Buscamos diamante
    - si diamante no rompe, sacamos lambda, pero hay que ver si es producto etc
    - si diamante rompe, probablemente sea por ks mayores a g, y la suma de los sobrepasos por encima de g del monomio son el numero de filas con ceros (simétrico)
    """
    r = 4
    g = 3
    finder = HodgeFinder(r, g)
    finder.m_hodge.draw_diamond()
    m = finder.find_motive()
    print(compare(get_motive_chow(finder.curve, r, 1), subs_chow_into_curve(m, finder.curve)))

    