import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import sympy as sp
from typing import Tuple
from tqdm import tqdm

from motives.core.lambda_ring_expr import LambdaRingExpr
from motives.grothendieck_motives import Lefschetz
from motives.grothendieck_motives.curves import Curve, CurveChow, Jacobian
from motives.grothendieck_motives.moduli.scheme import VectorBundleModuli


L = Lefschetz()


# non symbolic h1(X) means motive library's standard h1(X), as opposed to sym_lambda_chow, so as to have distinct λ powers be distinct free symbols


def compare(m1: LambdaRingExpr, m2: LambdaRingExpr) -> bool:
    res = m1.equals(m2)
    if res:
        return res
    return False

def motive_generic(curve: Curve, r: int, d: int) -> LambdaRingExpr:
    vector_bundle = VectorBundleModuli(curve, r, d)
    return vector_bundle._compute_motive_rkr(r, d)

def motive_generic_clean(curve: Curve, r: int, d: int) -> LambdaRingExpr:
    H = curve.curve_chow
    J = Jacobian(curve).get_lambda_var(1)
    return (motive_generic(curve, r, d)/J).expand().simplify().subs({H.get_lambda_var(i): sym_lambda_chow(H, i) for i in range(1, H.g+1)})

def motive_positive(curve: Curve, r: int, d: int) -> LambdaRingExpr:
    if r not in {2, 3}:
        raise Exception("wrong rank")
    jacobian = Jacobian(curve)
    vector_bundle = VectorBundleModuli(curve, r, d)
    if r == 2:
        motive = vector_bundle._compute_motive_rk2()
    elif r == 3:
        motive = vector_bundle._compute_motive_rk3()
    return motive.subs(jacobian, jacobian._et_repr).expand().simplify()

def sym_lambda(X: Curve, k: int) -> sp.Symbol:
    """
    compute λk(X) symbolically
    """
    if k < 0:
        return 0
    elif k == 0:
        return 1
    elif k == 1:
        return sp.Symbol(X.name)
    return sp.Symbol(f"λ{k}({X.name})")

def sym_lambda_chow(H: CurveChow, k: int) -> sp.Symbol:
    """
    compute λk(h1(X)) symbolically
    """
    L = Lefschetz()
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
    Compute λk(X) in terms of h1(X) symbolically
    """
    H = X.curve_chow
    return X.get_lambda_var(k).subs({H.get_lambda_var(i): sym_lambda_chow(H, i) for i in range(1, H.g+1)})

def sym_lambda_chow_in_X(X: Curve, k: int) -> LambdaRingExpr:
    """
    compute λk(h1(X)) in terms of X symbolically
    """
    g = X.g
    if k > 2*g:
        return 0
    return sym_lambda(X, k) - sym_lambda(X, k-1) - L*sym_lambda(X, k-1) + L*sym_lambda(X, k-2) 

def subs_X(motive: LambdaRingExpr, X: Curve) -> LambdaRingExpr:
    """
    substitue X into polynomial in h1(X) (input not symbolic in h1(X))
    """
    H = X.curve_chow
    subs = {H.get_lambda_var(k): sym_lambda_chow(X, k) for k in range(2*X.g+1)}
    return motive.subs(subs).expand()

def subs_H(motive: LambdaRingExpr, X: Curve) -> LambdaRingExpr:
    """
    substitue h1(X) into polynomial in X (output not symbolic in h1(X))
    """
    subs = {sym_lambda(X, k): X.get_lambda_var(k) for k in range(1, 2*X.g+1)}
    return motive.subs(subs).expand()

def gl(X: Curve, r: int) -> LambdaRingExpr:
    if r == 2:
        return gl_2(X)
    elif r == 3:
        return gl_3(X)
    return 0

def gl_2(X: Curve) -> LambdaRingExpr:
    g = X.g
    motive = sum(sym_lambda(X, k)*(L**k+L**(3*g-3-2*k)) for k in range(g-1)) + sym_lambda(X, g-1)*L**(g-1)
    return motive

def gl_3(X: Curve) -> LambdaRingExpr:
    g = X.g
    motive = sum(sym_lambda(X, a)*sym_lambda(X, b)*(L**(a+2*b)+L**(8*g-8-2*a-3*b)) for a in range(2*g-2) for b in range(2*g-2-a)) + sum(sym_lambda(X, a)*sym_lambda(X, (2*g-2-a))*(L**(a+2*(2*g-2-a))+L**(8*g-8-2*a-3*(2*g-2-a))) for a in range(g-1)) + sym_lambda(X, g-1)**2*L**(3*g-3)
    return motive

def biggest_term(expr: LambdaRingExpr) -> Tuple[tuple, LambdaRingExpr]:
    """
    computes vector of exponents of biggest monomial in expr together with it's coefficient, where expr ∈ Z[L][h1(X)]
    """
    lambdas = sorted(expr.free_symbols-{L}, key=lambda x: (len(x.name), x.name), reverse=True)
    if len(lambdas) == 0:
        return ((0,), expr)
    poly = sp.Poly(expr, *lambdas)
    return poly.terms()[0]

def find_motive_low(obj: LambdaRingExpr, X: Curve) -> LambdaRingExpr:
    """
    finds motive of obj in X, given the following conditions:
    - obj is a positive poynomial in h1(X)
    - it is known that obj has a positive decompostion in X that does not depend on λk(X) for k>g
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

def find_motive_bfs(obj: LambdaRingExpr, X: Curve, coefs: list[LambdaRingExpr], monoms_H: list[LambdaRingExpr], monoms_X: list[LambdaRingExpr], n_rounds: int) -> list[LambdaRingExpr]: 
    """
    compute all possible motivic decompositions given monomials and coefficients
    """
    candidates = {(obj, 0)}
    for n, monom in enumerate(monoms_H):
        for i in range(n_rounds):
            print(f"{n+1} monomial, {i+1} round, {len(candidates)} candidates")
            new_candidates = set()
            for candidate, big_part in tqdm(candidates):
                for coef in coefs:
                    m = (candidate - coef*monom).expand()
                    if not any(term.could_extract_minus_sign() for term in m.as_ordered_terms()):
                        new_candidates.add((m, big_part+coef*monoms_X[n]))
            candidates = new_candidates
    motives = {(big_part + m).expand() for candidate, big_part in tqdm(candidates) if (m := find_motive_low(candidate, X)) is not None}
    return list(motives)

def find_monomials(obj: LambdaRingExpr) -> Tuple[list[LambdaRingExpr], list[LambdaRingExpr]]:
    """
    Compute monomials of high degree that obj might depend on
    """
    pass