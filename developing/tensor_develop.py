from developing.utils import *
import numpy as np

"""
Vectorizamos todo
"""

class Vectorizer:

    def __init__(self, d: int, r: int, g: int) -> None:
        self.d = d
        self.r = r
        self.g = g
        self.curve = Curve("X", g)
        self.obj = symbolize_chow(get_motive_chow(self.curve, r, d), self.curve)
        self.lambdas = sorted(self.obj.free_symbols, key=lambda x: (len(x.name), x.name))
        self.monom_chow2idx = self.create_monom2idx(g+1, r, 7, 50) #rellenar bien
        self.idx2monom_chow = {idx: monom for monom, idx in self.monom_chow2idx.items()}
        self.monoms, self.monom2vec = self.create_monom2vec()
        self.matrix = self.get_matrix()
        self.vobj = self.expr2vec(obj)
        print("Prepared. Looking for motive.....")
        self.sol = self._solve()
        print("Finished!")
        self.motive = 0
        for i in range(len(self.sol)):
            v = self.monoms[i]
            self.motive += self.sol[i]*sp.prod([L**v[0]] + [sym_lambda(self.curve, i)**v[i] for i in range(1, len(v))])

    def expr2vec(self, expr: LambdaRingExpr):
        # turns expression in chow two vector, where a basis is given by monom_chow2idx
        expr = symbolize_chow(expr, self.curve)
        terms = self.get_terms(expr)
        vec = np.zeros(len(self.monom_chow2idx), dtype=int)
        for monom, coef in terms:
            vec[self.monom_chow2idx[monom]] = int(coef)
        return vec
    
    def create_monom2vec(self):
        tuples = list(self.create_monom2idx(2*g+1, r, (r-1)*g-2, (r**2-1)*(g-1)).keys())
        monom2vec = {v: self.expr2vec(symbolize_chow(sp.prod([L**v[0]] + [self.curve.get_lambda_var(i)**v[i] for i in range(1, len(v))]), self.curve)) for v in tuples}
        return tuples, monom2vec
 
    @staticmethod
    def create_monom2idx(n: int, k: int, p: int, q: int):
        """
        Return a dict mapping every n-tuple of non-negative integers to a unique index.

        Each tuple (x_0, x_1, ..., x_{n-1}) satisfies:
        - all x_i >= 0
        - at most k of (x_1, ..., x_{n-1}) are non-zero
        - sum(x_1, ..., x_{n-1}) <= p
        - sum(x_0, ..., x_{n-1}) <= q
        """
        results = []

        def gen(pos, tail_budget, total_budget, nonzero_used, current):
            if pos == n:
                results.append(tuple(current))
                return
            if pos == 0:
                for v in range(0, total_budget + 1):
                    gen(1, tail_budget, total_budget - v, nonzero_used, current + [v])
            else:
                budget = min(tail_budget, total_budget)
                for v in range(0, budget + 1):
                    if v > 0 and nonzero_used == k:
                        break
                    gen(pos + 1, tail_budget - v, total_budget - v,
                        nonzero_used + (v > 0), current + [v])

        gen(0, p, q, 0, [])
        return {t: i for i, t in enumerate(results)}
    
    def get_terms(self, expr: LambdaRingExpr) -> tuple[tuple, LambdaRingExpr]:
        """
        returns a list of vector of exponents together with their coefficient, where expr ∈ Z[L][h1(X)]
        """
        poly = sp.Poly(expr, *self.lambdas)
        return poly.terms()
    
    def get_matrix(self):
        return np.column_stack(list(self.monom2vec.values()))
    
    # ------------------------------ ILP -----------------------------------

    def _solve(self) -> np.ndarray | None:
        if not self._precheck():
            return None
        M = self.matrix.astype(np.float64)
        v = self.vobj.astype(np.float64)
        result = self._solve_lp(M, v)
        return result if result is not None else self._solve_ilp(M, v)

    def _precheck(self) -> bool:
        from math import gcd
        from functools import reduce
        for i in range(self.matrix.shape[0]):
            row_gcd = reduce(gcd, self.matrix[i].tolist())
            if row_gcd > 0 and self.vobj[i] % row_gcd != 0:
                return False
        return True

    def _solve_lp(self, M: np.ndarray, v: np.ndarray) -> np.ndarray | None:
        import highspy
        h = self._highs_with_constraints(M, v, integer=False)
        h.run()
        if h.getInfoValue("primal_solution_status")[1] != 2:
            return None
        x = np.round(np.array(h.getSolution().col_value)).astype(int)
        if np.all(x >= 0) and np.array_equal(self.matrix @ x, self.vobj):
            return x
        return None

    def _solve_ilp(self, M: np.ndarray, v: np.ndarray) -> np.ndarray | None:
        import highspy
        h = self._highs_with_constraints(M, v, integer=True)
        h.run()
        if h.getInfoValue("primal_solution_status")[1] != 2:
            return None
        return np.round(np.array(h.getSolution().col_value)).astype(int)

    def _highs_with_constraints(self, M: np.ndarray, v: np.ndarray, integer: bool):
        import highspy
        n = M.shape[1]
        h = highspy.Highs()
        h.silent()
        h.addVars(n, np.zeros(n), np.full(n, np.inf))
        if integer:
            h.changeColsIntegralityByRange(0, n - 1, [highspy.kInteger] * n)
        for i in range(M.shape[0]):
            row = M[i]
            nz = np.where(row != 0)[0]
            if len(nz) == 0:
                continue
            h.addRow(v[i], v[i], len(nz), nz.tolist(), row[nz].tolist())
        return h

if __name__ == "__main__":
    d = 1
    r = 4
    g = 3
    X = Curve("X", g)
    obj = get_motive_chow(X, r, d)
    vectorizer = Vectorizer(d, r, g)
    m = vectorizer.motive                      # True = non-negative
    print(compare(subs_chow_into_curve(m, vectorizer.curve), obj))
    print(m)