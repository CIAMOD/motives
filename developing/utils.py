import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import sympy as sp
import pickle
import os
from tqdm import tqdm

from motives.core.lambda_ring_expr import LambdaRingExpr
from motives.grothendieck_motives import Lefschetz
from motives.grothendieck_motives.curves import Curve, CurveChow, Jacobian
from motives.grothendieck_motives.moduli.scheme import VectorBundleModuli


L = Lefschetz()
EXPR_DIR = "expressions"


# non symbolic h1(X) means motive library's standard h1(X), as opposed to sym_lambda_chow, so as to have distinct λ powers be distinct free symbols


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

def sym_lambda_chow(H: CurveChow, k: int) -> sp.Symbol:
    """
    returns λk(h1(X)) symbolically
    """
    g = H.g
    if k < 0 or k > 2*g:
        return 0
    elif k == 0:
        return 1
    elif k <= g:
        return sp.Symbol(f"λ{k}(h1_{H.name})")
    return L**(k-g)*sp.Symbol(f"λ{2*g-k}(h1_{H.name})")  #no debería entrar aquí en lo que voy a usar

def sym_lambda_in_chow(X: Curve, k: int) -> LambdaRingExpr:
    """
    returns λk(X) in terms of h1(X) symbolically
    """
    H = X.curve_chow
    return X.get_lambda_var(k).subs({H.get_lambda_var(i): sym_lambda_chow(H, i) for i in range(1, H.g+1)})

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
    returns monomial in X in terms of h1(X) symbolically
    """
    return sp.prod(sym_lambda_in_chow(X, k) for k in exps)

def subs_curve(motive: LambdaRingExpr, X: Curve) -> LambdaRingExpr:
    """
    returns X into polynomial in h1(X) (input not symbolic in h1(X))
    """
    H = X.curve_chow
    subs = {H.get_lambda_var(k): sym_lambda_chow(X, k) for k in range(2*X.g+1)}
    return motive.subs(subs).expand()

def subs_chow(motive: LambdaRingExpr, X: Curve) -> LambdaRingExpr:
    """
    returns h1(X) into polynomial in X (output not symbolic in h1(X))
    """
    subs = {sym_lambda(X, k): X.get_lambda_var(k) for k in range(1, 2*X.g+1)}
    return motive.subs(subs).expand()

def save_expr(expr: LambdaRingExpr, file_name: str) -> None:
    """
    Saves expr to a file, substituting Lefschetz by string symbol due to class issue
    """
    with open(f"{EXPR_DIR}/{file_name}.pkl", "wb") as file:
        pickle.dump(expr.subs(L, sp.Symbol("L")), file)

def load_expr(file_name: str) -> LambdaRingExpr:
    """
    Saves expr to a file, substituting Lefschetz by string symbol due to class issue
    """
    with open(f"{EXPR_DIR}/{file_name}.pkl", "rb") as file:
        expr = pickle.load(file)
    return expr.subs(sp.Symbol("L"), L)

def motive_generic(curve: Curve, r: int, d: int) -> LambdaRingExpr:
    """
    returns the generic epxression of M(r, g), where g is the genus of curve, in terms of h1(X) non symbolically
    """
    vector_bundle = VectorBundleModuli(curve, r, d)
    return vector_bundle._compute_motive_rkr(r, d)

def motive_generic_clean(X: Curve, r: int, d: int) -> LambdaRingExpr:
    """
    returns the generic epxression of M(r, g) divided by Jacobian(X), where g is the genus of X, in terms of h1(X) symbolically 
    """
    H = X.curve_chow
    J = Jacobian(X).get_lambda_var(1)
    return (motive_generic(X, r, d)/J).expand().simplify().subs({H.get_lambda_var(i): sym_lambda_chow(H, i) for i in range(1, H.g+1)})

def get_motive_h1(X: Curve, r: int, d: int) -> LambdaRingExpr:
    """
    returns the generic epxression of M(r, g) divided by Jacobian(X), where g is the genus of X, in terms of h1(X) symbolically, with cache
    """
    file_name = f"{r}_{X.g}_h1"
    if os.path.exists(f"{EXPR_DIR}/{file_name}.pkl"):
        print("loaded from cache")
        return load_expr(file_name)
    print("calculating from scratch...")
    return motive_generic_clean(X, r, d)

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
    returns positive polynomial expression of M(r, g) in X, where g is the genus of curve and r∈{2,3}, symbolically
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
    return {k: [(sym_lambda_monomial(X, partition), sym_lambda_in_chow_monomial(X, partition)) for partition in get_partitions(k, min_first=g+1)] for k in range(g+1, (r-1)*(g-1)+1)}

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
    v, coef = biggest_term(obj)
    if all(i==0 for i in v):
        return coef
    monom_X = coef*sp.prod(sym_lambda(X, len(v)-k)**v[k] for k in range(len(v)))
    monom_H = coef*sp.prod(sym_lambda_in_chow(X, len(v)-k)**v[k] for k in range(len(v)))
    rest = (obj - monom_H).expand()
    rest_result = find_motive_low(rest, X)
    if rest_result != None:
        return monom_X + rest_result
    return None

def find_motive_bfs(obj: LambdaRingExpr, X: Curve, coefs: list[LambdaRingExpr], monoms, n_rounds: int, max_dim: int=-1) -> list[LambdaRingExpr]: 
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