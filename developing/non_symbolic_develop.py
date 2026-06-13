from developing.utils import *
from time import time

"""
--- VERSION NO SIMBOLICA ---
todo esto en h1(X):
(k,a1,....,ag) será L^k*λ1(X)^a1*...*λg(X)^ag
sumas de esto serán listas de tuplas
"""

# region 
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
    if len(expr_sym.free_symbols) == 0:
        data = {(0 for _ in range(g+1)): 1}
        return NExpr(data, g+1)
    terms = sp.Poly(expr_sym, *sorted(expr_sym.free_symbols, key=lambda x: (len(x.name), x.name))).terms()
    n_pad = g+1 - len(terms[0][0])
    if n_pad > 0:
        for i in range(len(terms)):
            terms[i] = (terms[i][0] + n_pad*(0,), terms[i][1])
    data = {monom: coef for monom, coef in terms}
    return NExpr(data, g+1)

def get_s2ns(X: Curve, r: int) -> dict[LambdaRingExpr, NExpr]:
    g = X.g
    monoms = {k: [(sym_lambda_monomial(X, partition), sym_lambda_in_chow_monomial(X, partition)) for partition in get_partitions(k)] for k in range(1, (r-1)*g-2+1)} #ajustar
    s2ns = {L**i*monom_x: expr_to_nexpr(L**i*symbolize_chow(monom_h1, X), X) for k in monoms.keys() for monom_x, monom_h1 in monoms[k] for i in range(max((r**2-1)*(g-1)+1, g+1))} #ajustar
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

def is_valid(expr: NExpr):
    valid = all(c>0 for c in expr.data.values())
    return valid

def ndfs(expr_x: LambdaRingExpr, expr_h1: NExpr, X: Curve, monoms: list, s2ns, coefs, n_rounds, monom_count, start_monoms: int = 0, start_coefs: int = 0, depth: int = 0) -> LambdaRingExpr:    
    for i in range(start_monoms, len(monoms)):
        if monom_count[i] < n_rounds:
            k = start_coefs if i==start_monoms else 0
            for j in range(k, len(coefs)):
                coef = coefs[j]
                monom_x = monoms[i] * coef
                monom_h1 = s2ns[monom_x]
                new_expr_h1 = expr_h1 - monom_h1
                if is_valid(new_expr_h1):
                    new_expr_x = expr_x + monom_x
                    monom_count[i] += 1
                    candidate = ndfs(new_expr_x, new_expr_h1, X, monoms, s2ns, coefs, n_rounds, monom_count, i, j, depth+1)  
                    monom_count[i] -= 1
                    if candidate is not None:
                        return candidate
    candidate = nfind_motive_low(expr_h1, X, s2ns)
    return expr_x + candidate if candidate is not None else None

def nbfs(obj: NExpr, X: Curve, monoms: dict, s2ns, coefs: list[LambdaRingExpr], n_rounds: int, max_dim: int=-1) -> set[LambdaRingExpr]: 
    candidates = {(obj, 0)}
    for degree in sorted(monoms.keys(), reverse=True):
        highest_coef_degree = len(coefs) if max_dim == -1 else max(max_dim-degree+1, 0) # cut coefs (revisar?)
        for monom_X_pure, _ in monoms[degree]:
            for i in range(n_rounds):
                # print(f"{degree} degree monomial, {i+1} round, {len(candidates)} candidates")
                new_candidates = set()
                for candidate, big_part in tqdm(candidates):
                    for coef in coefs[:highest_coef_degree]:   # cut coefs 
                        monom_X = coef*monom_X_pure
                        monom_H = s2ns[monom_X]
                        m = candidate - monom_H
                        if all(c>0 for c in obj.data.values()):
                            new_candidates.add((m, big_part+monom_X))    # revisar unicidad de motivos
                candidates = candidates.union(new_candidates)
    motives = {(big_part + m).expand() for candidate, big_part in tqdm(candidates) if (m := nfind_motive_low(candidate, X, s2ns)) is not None}
    return motives

def find_motive(obj, X, r):
    nobj = expr_to_nexpr(obj, X)
    coefs = get_coefficients(r**2*(X.g-1)-2*X.g)
    monoms = get_small_monomials(X, r)
    print(monoms)
    s2ns = get_s2ns(X, r)
    print(s2ns.keys())
    print("done")
    return nbfs(nobj, X, monoms, s2ns, coefs, 2)
# endregion

class MotiveFinder:

    def __init__(self, r: int, g: int, d: int):
        self.r = r
        self.g = g
        self.d = d
        self.X = Curve("X", g)
        self.Jac = Jacobian(self.X)
        self.low_motives = {i: get_motive_chow(self.X, i, d) for i in range(2, r)}
        self.pieces = self.get_pieces()
        self.coefs = self.get_coefficients()
        self.s2ns = self.get_s2ns()
        print(f"MotiveFinder({r},{g},{d}) initialized")

    def nfind_motive_low(self, obj: NExpr) -> LambdaRingExpr:
        if any(coef<0 for coef in obj.data.values()):
            return None
        if len(obj.data.items()) == 0:
            return 0
        v, coef = max(obj.data.items(), key=lambda x: x[0][::-1])
        if all(i==0 for i in v[1:]):
            return obj.to_sympy(self.X)
        monom_x = L**(v[0])*sp.prod(sym_lambda(self.X, k)**v[k] for k in range(1, len(v)))
        monom_h1 = self.s2ns[monom_x]
        rest = obj - monom_h1*coef
        rest_result = self.nfind_motive_low(rest)
        if rest_result != None:
            return coef*monom_x + rest_result
        return None

    def ndfs(self, expr_x: LambdaRingExpr, expr_h1: NExpr, start_monoms: int = 0, start_coefs: int = 0) -> LambdaRingExpr:    
        for i in range(start_monoms, len(self.pieces)):
            k = start_coefs if i==start_monoms else 0
            for j in range(k, len(self.coefs)):
                coef = self.coefs[j]
                monom_x = self.pieces[i] * coef
                monom_h1 = self.s2ns[monom_x]
                new_expr_h1 = expr_h1 - monom_h1
                if is_valid(new_expr_h1):
                    new_expr_x = expr_x + monom_x
                    candidate = self.ndfs(new_expr_x, new_expr_h1, i, j)  
                    if candidate is not None:
                        return candidate
        candidate = self.nfind_motive_low(expr_h1)
        return expr_x + candidate if candidate is not None else None

    def find_motive(self, obj):   # un poco raro que obj sea parámetro pero luego lo cambiamos
        nobj = expr_to_nexpr(obj, self.X)
        return self.ndfs(0, nobj)
    
    def get_motive_chow(self):
        return symbolize_chow(get_motive_chow(self.X, self.r, self.d), self.X)
    
    def get_coefficients(self):
        return [L**i for i in range(25)][::-1]
    
    def get_pieces(self):
        J = sp.Symbol("Jac(X)")
        bound = max(self.g, ((self.r-1)*self.g-2))
        pieces = [J**i*sym_lambda_monomial(self.X, partition) for k in range(bound+1) for partition in get_partitions(k, max_first=self.g) for i in range(1,2)]
        for i in range(2,self.r):
            M_low = sp.Symbol(f"M({i},L)")
            pieces += [M_low*sym_lambda_monomial(self.X, partition) for k in range(bound+1) for partition in get_partitions(k, max_first=self.g)]
        return pieces[::-1]

    def get_s2ns(self):
        J = sp.Symbol("Jac(X)")
        bound = max(self.g, ((self.r-1)*self.g-2))
        s2ns = {L**j*J**i*sym_lambda_monomial(self.X, partition): expr_to_nexpr((L**j*self.Jac.to_lambda()**i*sym_lambda_in_chow_monomial(self.X, partition)).expand(), self.X) for k in range(bound+1) for partition in get_partitions(k, max_first=self.g) for i in range(2) for j in range(25)}  #el 19 es temporal
        for i in range(2,self.r):
            M_low = sp.Symbol(f"M({i},L)")
            s2ns |= {L**j*M_low*sym_lambda_monomial(self.X, partition): expr_to_nexpr((L**j*self.low_motives[i]*sym_lambda_in_chow_monomial(self.X, partition)).expand(), self.X) for k in range(bound+1) for partition in get_partitions(k, max_first=self.g) for j in range(25)}
        return s2ns

class MotiveFinderClassic:

    def __init__(self, r: int, g: int, d: int):
        self.r = r
        self.g = g
        self.d = d
        self.X = Curve("X", g)
        self.pieces = self.get_pieces()
        self.coefs = self.get_coefficients()
        self.s2ns = self.get_s2ns()
        print(f"MotiveFinder({r},{g},{d}) initialized")

    def nfind_motive_low(self, obj: NExpr) -> LambdaRingExpr:
        if any(coef<0 for coef in obj.data.values()):
            return None
        if len(obj.data.items()) == 0:
            return 0
        v, coef = max(obj.data.items(), key=lambda x: x[0][::-1])
        if all(i==0 for i in v[1:]):
            return obj.to_sympy(self.X)
        monom_x = L**(v[0])*sp.prod(sym_lambda(self.X, k)**v[k] for k in range(1, len(v)))
        monom_h1 = self.s2ns[monom_x]
        rest = obj - monom_h1*coef
        rest_result = self.nfind_motive_low(rest)
        if rest_result != None:
            return coef*monom_x + rest_result
        return None

    def ndfs(self, expr_x: LambdaRingExpr, expr_h1: NExpr, start_monoms: int = 0, start_coefs: int = 0) -> LambdaRingExpr:    
        for i in range(start_monoms, len(self.pieces)):
            k = start_coefs if i==start_monoms else 0
            for j in range(k, len(self.coefs)):
                coef = self.coefs[j]
                monom_x = self.pieces[i] * coef
                monom_h1 = self.s2ns[monom_x]
                new_expr_h1 = expr_h1 - monom_h1
                if is_valid(new_expr_h1):
                    new_expr_x = expr_x + monom_x
                    candidate = self.ndfs(new_expr_x, new_expr_h1, i, j)  
                    if candidate is not None:
                        return candidate
        candidate = self.nfind_motive_low(expr_h1)
        return expr_x + candidate if candidate is not None else None

    def find_motive(self, obj):   # un poco raro que obj sea parámetro pero luego lo cambiamos
        nobj = expr_to_nexpr(obj, self.X)
        return self.ndfs(0, nobj)
    
    def get_motive_chow(self):
        return get_motive_chow(self.X, self.r, self.d)
    
    def get_coefficients(self):
        bound = (self.r**2-3)*self.g - (self.r**2+1)
        return [L**i for i in range(bound+1)][::-1]
    
    def get_pieces(self):
        monoms = get_small_monomials(self.X, self.r)
        pieces = [item[0] for degree in monoms.keys() for item in monoms[degree]]
        return pieces[::-1]   

    def get_s2ns(self):
        bound = max(self.g, ((self.r-1)*self.g-2))
        s2ns = {L**j*sym_lambda_monomial(self.X, partition): expr_to_nexpr((L**j*sym_lambda_in_chow_monomial(self.X, partition)).expand(), self.X) for k in range(bound+1) for partition in get_partitions(k) for j in range(25)}  #el 19 es temporal
        return s2ns

class ModuliFinder:

    def __init__(self, r: int, g: int, d: int):
        self.r = r
        self.g = g
        self.d = d
        self.X = Curve("X", g)
        self.pieces = self.get_pieces()
        self.s2ns = self.get_s2ns()
        print(f"MotiveFinder({r},{g},{d}) initialized")

    def nfind_motive_low(self, obj: NExpr) -> LambdaRingExpr:
        if any(coef<0 for coef in obj.data.values()):
            return None
        if len(obj.data.items()) == 0:
            return 0
        v, coef = max(obj.data.items(), key=lambda x: x[0][::-1])
        if all(i==0 for i in v[1:]):
            return obj.to_sympy(self.X)
        monom_x = L**(v[0])*sp.prod(sym_lambda(self.X, k)**v[k] for k in range(1, len(v)))
        monom_h1 = self.s2ns[monom_x]
        rest = obj - monom_h1*coef
        rest_result = self.nfind_motive_low(rest)
        if rest_result != None:
            return coef*monom_x + rest_result
        return None

    def ndfs(self, expr_x: LambdaRingExpr, expr_h1: NExpr, start_idx=0) -> LambdaRingExpr:    
        for i in range(start_idx, len(self.pieces)):
            monom_x = self.pieces[i]
            monom_h1 = self.s2ns[monom_x]
            new_expr_h1 = expr_h1 - monom_h1
            if is_valid(new_expr_h1):
                new_expr_x = expr_x + monom_x
                candidate = self.ndfs(new_expr_x, new_expr_h1, i)  
                if candidate is not None:
                    return candidate
        candidate = self.nfind_motive_low(expr_h1)
        return expr_x + candidate if candidate is not None else None

    def find_motive(self, obj):   # un poco raro que obj sea parámetro pero luego lo cambiamos
        nobj = expr_to_nexpr(obj, self.X)
        return self.ndfs(0, nobj)
    
    def get_motive_chow(self):
        return get_motive_chow(self.X, self.r, self.d)
    
    def get_pieces(self):
        monoms = get_small_monomials(self.X, self.r)
        top = (r**2-3)*g - (r**2+1)
        pieces = [L**i*item[0] for i in range(top, -1, -1) for degree in sorted(monoms.keys(), reverse=True) for item in monoms[degree]]
        return pieces

    def get_s2ns(self):
        bound = max(self.g, ((self.r-1)*self.g-2))
        s2ns = {L**j*sym_lambda_monomial(self.X, partition): expr_to_nexpr((L**j*sym_lambda_in_chow_monomial(self.X, partition)).expand(), self.X) for k in range(bound+1) for partition in get_partitions(k) for j in range(25)}  #el 19 es temporal
        return s2ns

def save_expr(exprs: LambdaRingExpr, X: Curve, file_name: str) -> None:
    """
    saves expr to a file, substituting Lefschetz by string symbol due to class issue
    """
    sym_exprs = [symbolize_chow(expr, X).subs(L, sp.Symbol("L")) for expr in exprs]
    with open(f"{EXPR_DIR}/{file_name}_multiple.pkl", "wb") as file:
        pickle.dump(sym_exprs, file)

def partial_gl4(X: Curve):
    g = X.g
    top = 3
    return sum(X.get_lambda_var(k1)*X.get_lambda_var(k2)*X.get_lambda_var(k3)*(L**(k1+2*k2+3*k3)+L**(15*g-15-2*k1-3*k2-4*k3)) for k1 in range(top) for k2 in range(top-k1) for k3 in range(top-k1-k2)) + X.get_lambda_var(g-1)**3*L**(6*(g-1))


if __name__ == "__main__":
    r = 4
    g = 2
    d = 1
    finder = MotiveFinder(r,g,d)
    print(finder.find_motive(finder.get_motive_chow()))
