import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import os
import pickle
from tqdm import tqdm
import sympy as sp

from motives.core.lambda_ring_expr import LambdaRingExpr
from motives.grothendieck_motives import Lefschetz
from motives.grothendieck_motives.curves import Curve, Jacobian, CurveChow
from motives.grothendieck_motives.moduli.scheme import VectorBundleModuli


L = Lefschetz()
EXPR_DIR = "developing/expressions"

def symbolize_chow(expr: LambdaRingExpr, X: Curve):
    """
    returns expression with symbolyzed h1(X)
    """
    H = X.curve_chow
    subs1 = {H.get_lambda_var(k): sp.Symbol(f"λ{k}(h1_{H.name})") for k in range(2, X.g+1)}
    subs2 = {H.get_lambda_var(1): sp.Symbol(f"λ{1}(h1_{H.name})")}
    return expr.subs(subs1).subs(subs2)

def desymbolize_chow(expr: LambdaRingExpr, X: Curve):
    """
    returns expression with desymbolyzed h1(X)
    """
    H = X.curve_chow
    return expr.subs({sp.Symbol(f"λ{k}(h1_{H.name})"): H.get_lambda_var(k) for k in range(1, X.g+1)})

def compare(m1: LambdaRingExpr, m2: LambdaRingExpr) -> bool:
    """
    returns True if expressions are equal, False if they are unequal or it is unknown
    """
    res = m1.equals(m2)
    if res:
        return res
    return False

def sym_lambda(X: Curve, k: int) -> sp.Symbol:
    """
    returns λk(X) symbolically
    """
    if k < 0:
        return 0
    elif k == 0:
        return 1
    return sp.Symbol(f"λ{k}({X.name})")

def sym_lambda_in_chow(X: Curve, k: int) -> sp.Symbol:
    """
    returns λk(X) in terms of h1(X) symbolically
    """
    return symbolize_chow(X.get_lambda_var(k), X)

def sym_lambda_chow_in_chow(X: Curve, k: int) -> sp.Symbol:
    """
    returns λk(h1(X)) in terms of h1(X) symbolically
    """
    if k <= X.g:
        return symbolize_chow(X.curve_chow.get_lambda_var(k), X)
    elif k <= 2*X.g:
        i = k-X.g
        return L**i*symbolize_chow(X.curve_chow.get_lambda_var(X.g-i), X)
    return 0

def sym_lambda_chow_in_curve(X: Curve, k: int) -> LambdaRingExpr:
    """
    returns λk(h1(X)) in terms of X symbolically
    """
    g = X.g
    if k > 2*g:
        return 0
    return sym_lambda(X, k) - sym_lambda(X, k-1) - L*sym_lambda(X, k-1) + L*sym_lambda(X, k-2) 

def sym_lambda_monomial(X: Curve, exps: list[int]) -> LambdaRingExpr:
    """
    returns monomial in X symbolically
    """
    return sp.prod(sym_lambda(X, k) for k in exps)

def sym_lambda_in_chow_monomial(X: Curve, exps: list[int]) -> LambdaRingExpr:
    """
    returns monomial in X in terms of h1(X)
    """
    return sp.prod(sym_lambda_in_chow(X, k) for k in exps)

def subs_chow_into_curve(expr: LambdaRingExpr, X: Curve) -> LambdaRingExpr:
    """
    returns expr in terms of h1(X)
    """
    subs = {sym_lambda(X, k): X.get_lambda_var(k) for k in range(1, 2*X.g+1)}
    return expr.subs(subs).expand()

def save_expr(expr: LambdaRingExpr, X: Curve, file_name: str) -> None:
    """
    saves expr to a file, substituting Lefschetz by string symbol due to class issue
    """
    sym_expr = symbolize_chow(expr, X).subs(L, sp.Symbol("L"))
    with open(f"{EXPR_DIR}/{file_name}.pkl", "wb") as file:
        pickle.dump(sym_expr, file)

def load_expr(X: Curve, file_name: str) -> LambdaRingExpr:
    """
    loads expression
    """
    with open(f"{EXPR_DIR}/{file_name}.pkl", "rb") as file:
        sym_expr = pickle.load(file)
    expr = desymbolize_chow(sym_expr, X).subs(sp.Symbol("L"), L)
    return expr

def motive_generic(curve: Curve, r: int, d: int) -> LambdaRingExpr:
    """
    returns the generic epxression of M(r, g), where g is the genus of curve, in terms of h1(X) 
    """
    vector_bundle = VectorBundleModuli(curve, r, d)
    return vector_bundle._compute_motive_rkr(r, d)

def motive_generic_divided(X: Curve, r: int, d: int) -> LambdaRingExpr:
    """
    returns the generic epxression of M(r, g) divided by Jacobian(X), where g is the genus of X, in terms of h1(X)
    """
    J = Jacobian(X).get_lambda_var(1)
    return (motive_generic(X, r, d)/J).expand().simplify()

def get_motive_chow(X: Curve, r: int, d: int) -> LambdaRingExpr:
    """
    returns the generic epxression of M(r, g) divided by Jacobian(X), where g is the genus of X, in terms of h1(X)
    """
    file_name = f"{r}_{X.g}_h1"
    if os.path.exists(f"{EXPR_DIR}/{file_name}.pkl"):
        print("loaded from cache")
        return load_expr(X, file_name)
    print("calculating from scratch...")
    return motive_generic_divided(X, r, d)

def gl_2(X: Curve) -> LambdaRingExpr:
    """
    returns positive polynomial expression in X of M(2, g), where g is the genus of curve, symbolically
    """
    g = X.g
    motive = sum(sym_lambda(X, k)*(L**k+L**(3*g-3-2*k)) for k in range(g-1)) + sym_lambda(X, g-1)*L**(g-1)
    return motive

def gl_3(X: Curve) -> LambdaRingExpr:
    """
    returns positive polynomial expression in X of M(3, g), where g is the genus of curve, symbolically
    """
    g = X.g
    motive = sum(sym_lambda(X, a)*sym_lambda(X, b)*(L**(a+2*b)+L**(8*g-8-2*a-3*b)) for a in range(2*g-2) for b in range(2*g-2-a)) + sum(sym_lambda(X, a)*sym_lambda(X, (2*g-2-a))*(L**(a+2*(2*g-2-a))+L**(8*g-8-2*a-3*(2*g-2-a))) for a in range(g-1)) + sym_lambda(X, g-1)**2*L**(3*g-3)
    return motive

def gl(X: Curve, r: int) -> LambdaRingExpr:
    """
    returns positive polynomial expression of M(r, g) in X, where g is the genus of curve and r ∈ {2,3}, symbolically
    """
    if r == 2:
        return gl_2(X)
    elif r == 3:
        return gl_3(X)
    raise Exception("r should be 2 or 3")

def biggest_term(expr: LambdaRingExpr) -> tuple[tuple, LambdaRingExpr]:
    """
    returns vector of exponents of biggest monomial in expr together with it's coefficient, where expr ∈ Z[L][h1(X)]
    """
    lambdas = sorted(expr.free_symbols-{L}, key=lambda x: (len(x.name), x.name), reverse=True)
    if len(lambdas) == 0:
        return (0,), expr
    poly = sp.Poly(expr, *lambdas)
    return poly.terms()[0]

def get_partitions(n: int, max_first: int=-1, min_first: int=1) -> list[list[int]]:
    """
    returns non increasing ordered partitions of n
    """
    if max_first == -1:
        max_first = n
    if n == 0:
        return [[]]
    return [[k] + partition for k in range(min(max_first,n), min_first-1, -1) for partition in get_partitions(n-k, max_first=k)]

def get_monomials(X: Curve, r: int) -> dict[int, list[tuple[LambdaRingExpr, LambdaRingExpr]]]:
    """
    returns all possible high degree monomials of X
    """
    g = X.g
    return {k: [(sym_lambda_monomial(X, partition), sym_lambda_in_chow_monomial(X, partition)) for partition in get_partitions(k, min_first=g+1)] for k in range(g+1, (r**2-1)*(g-1)+1)}

def get_small_monomials(X: Curve, r: int) -> dict[int, list[tuple[LambdaRingExpr, LambdaRingExpr]]]:
    """
    returns high degree monomials of X belonging to conjectured expression generalizing Gomez & Lee
    """
    g = X.g
    return {k: [(sym_lambda_monomial(X, partition), sym_lambda_in_chow_monomial(X, partition)) for partition in get_partitions(k, min_first=g+1)] for k in range(g+1, ((r-1)*g-2)+1)}

def get_coefficients(max_dim: int) -> list[LambdaRingExpr]:
    """
    returns lefschetz coefficients from exponent 0 to max_dim
    """
    return [L**i for i in range(max_dim+1)]

def find_motive_low(obj: LambdaRingExpr, X: Curve) -> LambdaRingExpr:
    """
    returns positive decomposition of obj in X, given the following conditions (if not, None may be returned):
    - obj is a positive poynomial in h1(X)
    - it is known that obj has a positive decomposition in X that does not depend on λk(X) for k>g
    """
    if any(term.could_extract_minus_sign() for term in obj.as_ordered_terms()):
        return None
    v, coef = biggest_term(symbolize_chow(obj, X))
    if all(i==0 for i in v):
        return coef
    monom_X = coef*sp.prod(sym_lambda(X, len(v)-k)**v[k] for k in range(len(v)))
    monom_H = coef*sp.prod(X.get_lambda_var(len(v)-k)**v[k] for k in range(len(v)))
    rest = (obj - monom_H).expand()
    rest_result = find_motive_low(rest, X)
    if rest_result != None:
        return monom_X + rest_result
    return None

def find_motives_bfs(obj: LambdaRingExpr, X: Curve, coefs: list[LambdaRingExpr], monoms, n_rounds: int, max_dim: int=-1) -> set[LambdaRingExpr]: 
    """
    returns all possible motivic decompositions given monomials (high degree) and coefficients
    """
    candidates = {(obj, 0)}
    for degree in sorted(monoms.keys(), reverse=True):
        highest_coef_degree = len(coefs) if max_dim == -1 else max(max_dim-degree+1, 0) # cut coefs (revisar?)
        for monom_X, monom_H in tqdm(monoms[degree]):
            for i in range(n_rounds):
                # print(f"{degree} degree monomial, {i+1} round, {len(candidates)} candidates")
                new_candidates = set()
                for candidate, big_part in candidates:
                    for coef in coefs[:highest_coef_degree]:   # cut coefs 
                        m = (candidate - coef*monom_H).expand()
                        if not any(term.could_extract_minus_sign() for term in m.as_ordered_terms()):
                            new_candidates.add((m, big_part+coef*monom_X))    # revisar unicidad de motivos
                candidates = candidates.union(new_candidates)
    motives = {(big_part + m).expand() for candidate, big_part in tqdm(candidates) if (m := find_motive_low(candidate, X)) is not None}
    return motives

# hay un orden mas eficiente? Igual por dimension total con L incluido? probado y parece que no

def get_terms(expr: LambdaRingExpr) -> tuple[tuple, LambdaRingExpr]:
    lambdas = sorted(expr.free_symbols-{L}, key=lambda x: (len(x.name), x.name), reverse=True)
    if len(lambdas) == 0:
        return (0,), expr
    return sp.Poly(expr, *lambdas).terms()


if __name__ == "__main__":
    g = 3
    k = 7
    curve = Curve("X", g)
    chow = curve.curve_chow
    print(sym_lambda_chow_in_chow(curve, k))
