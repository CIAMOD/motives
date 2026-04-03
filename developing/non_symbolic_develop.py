from developing.utils import *
from time import time

"""
--- VERSION NO SIMBOLICA ---
todo esto en h1(X):
(k,a1,....,ag) será L^k*λ1(X)^a1*...*λg(X)^ag
sumas de esto serán listas de tuplas
"""

MIN_DEPTH = 0
iteration_time = 0
iteration = 0

class NExpr:
    def __init__(self, data: dict, n: int) -> None:
        self.n = n
        self.data = data  # {(2,1,0): coef, ...}
    
    def __add__(self, other: 'NExpr') -> 'NExpr':
        result = self.data.copy()
        for tup, coef in other.data.items():
            result[tup] = result.get(tup, 0) + coef
            if result[tup] == 0:
                del result[tup]
        return NExpr(result, self.n)
    
    def __sub__(self, other: 'NExpr') -> 'NExpr':
        result = self.data.copy()
        for tup, coef in other.data.items():
            result[tup] = result.get(tup, 0) - coef
            if result[tup] == 0:
                del result[tup]
        return NExpr(result, self.n)
    
    def __mul__(self, other: int) -> 'NExpr':
        if other == 0:
            return NExpr({}, self.n)
        return NExpr({k: v * other for k, v in self.data.items() if v * other != 0}, self.n)
    
    def __repr__(self) -> None:
        return f"NExpr(n={self.n}, {self.data})"
    
    def to_sympy(self, X: Curve, chow=True) -> LambdaRingExpr:
        if not chow:
            raise ValueError("No chow not yet implemented")
        H = X.curve_chow
        expr = 0
        for v, coef in self.data.items():
            expr +=  coef*L**(v[0])*sp.prod(H.get_lambda_var(k)**v[k] for k in range(1, len(v)))
        return symbolize_chow(expr, X)


def expr_to_nexpr(expr: LambdaRingExpr, X: Curve) -> NExpr:
    expr_sym = symbolize_chow(expr, X)
    g = X.g
    terms = sp.Poly(expr_sym, *sorted(expr_sym.free_symbols, key=lambda x: (len(x.name), x.name))).terms()
    n_pad = g+1 - len(terms[0][0])
    if n_pad > 0:
        for i in range(len(terms)):
            terms[i] = (terms[i][0] + n_pad*(0,), terms[i][1])
    data = {monom: coef for monom, coef in terms}
    return NExpr(data, g+1)

def get_s2ns(X: Curve, r: int) -> dict[LambdaRingExpr, NExpr]:
    g = X.g
    monoms = {k: [(sym_lambda_monomial(X, partition), sym_lambda_in_chow_monomial(X, partition)) for partition in get_partitions(k)] for k in range(1, max(((r-1)*g-2)+1, g+1))}
    s2ns = {L**i*monom_x: expr_to_nexpr(L**i*symbolize_chow(monom_h1, X), X) for k in monoms.keys() for monom_x, monom_h1 in monoms[k] for i in range(max((r**2-1)*(g-1)+1, g+1)-k)}
    return s2ns

def nfind_motive_low(obj: NExpr, X: Curve, s2ns: dict[LambdaRingExpr, NExpr]) -> LambdaRingExpr:
    if any(coef<0 for coef in obj.data.values()):
        return None
    v, coef = max(obj.data.items(), key=lambda x: x[0][::-1])
    if all(i==0 for i in v[1:]):
        return obj.to_sympy(X)
    monom_x = L**(v[0])*sp.prod(sym_lambda(X, k)**v[k] for k in range(1, len(v)))
    monom_h1 = s2ns[monom_x]
    rest = obj - monom_h1*coef
    rest_result = nfind_motive_low(rest, X, s2ns)
    if rest_result != None:
        return coef*monom_x + rest_result
    return None

def ndfs(expr_x: LambdaRingExpr, expr_h1: NExpr, X: Curve, monoms: list, s2ns, coefs, n_rounds, monom_count, start_monoms: int = 0, start_coefs: int = 0, depth: int = 0) -> LambdaRingExpr:
    global iteration_time, iteration

    iteration += 1
    if iteration % 10000 == 0:
        print(f"{iteration} iterations done ({(iteration/305809*100):.2f}%)")
    candidate = nfind_motive_low(expr_h1, X, s2ns)
    if candidate is not None:
        return expr_x + candidate
    for i in range(start_monoms, len(monoms)):
        if monom_count[i] < n_rounds:
            k = start_coefs if i==start_monoms else 0
            for j in range(k, len(coefs)):
                coef = coefs[j]
                monom_x = monoms[i] * coef
                monom_h1 = s2ns[monom_x]
                new_expr_h1 = expr_h1 - monom_h1
                if all(c>0 for c in new_expr_h1.data.values()):
                    new_expr_x = expr_x + monom_x
                    monom_count[i] += 1
                    candidate = ndfs(new_expr_x, new_expr_h1, X, monoms, s2ns, coefs, n_rounds, monom_count, i, j, depth+1)  
                    monom_count[i] -= 1
                    if candidate is not None:
                        return candidate
    return None


if __name__ == "__main__":
    d = 1
    r = 4
    g = 3
    X = Curve("X", g)
    obj = get_motive_chow(X, r, d)
    monoms = get_small_monomials(X, r)
    monoms_list = [item[0] for sublist in reversed(monoms.values()) for item in sublist]
    monoms_list = [monoms_list[i] for i in (0,2)]
    s2ns = get_s2ns(X, r)
    coefs = list(reversed(get_coefficients((r**2-3)*g - (r**2+1))))
    t1 = time()
    print(monoms_list)
    result = ndfs(0, expr_to_nexpr(obj, X), X, monoms_list, s2ns, coefs, 2, [0 for _ in range(len(monoms_list))])
    print(time()-t1)
    print(result)
    save_expr(result, X, f"{r}_{g}")

