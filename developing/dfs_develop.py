# coeficiente L máximo en monomio !=1: (r**2-1)*g - (r**2+1) (hipotesis, comprobado en todo monomio computado)
# coeficiente L máximo en monomio con lambda>g: (r**2-3)*g - (r**2+1) (hipotesis debil, combrobado en todo monomio computado)

from developing.utils import *
from time import time

MIN_DEPTH = 2

def get_terms(expr: LambdaRingExpr) -> tuple[tuple, LambdaRingExpr]:
    lambdas = sorted(expr.free_symbols-{L}, key=lambda x: (len(x.name), x.name), reverse=True)
    if len(lambdas) == 0:
        return (0,), expr
    return sp.Poly(expr, *lambdas).terms()

def get_maximal_terms(expr: LambdaRingExpr) -> list[tuple[tuple, LambdaRingExpr]]:
    # see notepad
    terms = get_terms(expr)
    maximal_terms = []
    for v, coef in terms:
        maximal = True
        for w, _ in terms:
            if v != w and all(sum(v[:k])<=sum(w[:k]) for k in range(1, len(v)+1)):
                maximal = False
                break
        if maximal:
            maximal_terms.append((v, coef))
    return maximal_terms

def check_monom(v, coef):
    return all(c in (0, 1) for c in sp.Poly(coef, L).all_coeffs()) and not all(i==0 for i in v[1:])

def reduce_motive(obj: LambdaRingExpr, X: Curve, big_part: LambdaRingExpr) -> tuple[LambdaRingExpr, LambdaRingExpr]:
    reduceable_monoms = [(v, coef) for v, coef in get_maximal_terms(symbolize_chow(obj, X)) if check_monom(v, coef)]
    if len(reduceable_monoms) > 0:
        reduce_X = sum(coef*sp.prod(sym_lambda(X, len(v)-k)**v[k] for k in range(len(v))) for v, coef in reduceable_monoms)
        reduce_H = sum(coef*sp.prod(X.get_lambda_var(len(v)-k)**v[k] for k in range(len(v))) for v, coef in reduceable_monoms)
        rest = (obj - reduce_H).expand()
        return reduce_motive(rest, X, reduce_X+big_part)
    return obj, big_part

iteration = 0

def dfs(
    expr_x: LambdaRingExpr,
    expr_h1: LambdaRingExpr,
    X: Curve,
    monoms: list,
    coefs,
    n_rounds,
    monom_count,
    start_monoms: int = 0,
    start_coefs: int = 0,
    depth: int = 0,
) -> LambdaRingExpr:
    global iteration

    print(iteration)

    iteration += 1
    if depth >= MIN_DEPTH:
        candidate = find_motive_low(expr_h1, X)
        if candidate is not None:
            return expr_x + candidate

    expr_h1, expr_x = reduce_motive(expr_h1, X, expr_x)
    for i in range(start_monoms, len(monoms)):
        if monom_count[i] < n_rounds:
            k = start_coefs if i==start_monoms else 0
            for j in range(k, len(coefs)):
                monom_x, monom_h1 = monoms[i]
                coef = coefs[j]
                new_expr_h1 = expr_h1 - (coef * monom_h1).expand()
                if not any(term.could_extract_minus_sign() for term in new_expr_h1.as_ordered_terms()):
                    new_expr_x = expr_x + coef * monom_x
                    monom_count[i] += 1
                    candidate = dfs(new_expr_x, new_expr_h1, X, monoms, coefs, n_rounds, monom_count, i, j, depth+1)  
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
    monoms_list = [item for sublist in reversed(monoms.values()) for item in sublist]
    coefs = get_coefficients((r**2-3)*g - (r**2+1))
    t1 = time()
    result = dfs(0, obj, X, monoms_list, coefs, 2, [0 for _ in range(len(monoms_list))])
    print(f"{time()-t1}s")
    print(result.expand())
    print(compare(subs_chow_into_curve(result, X), obj))