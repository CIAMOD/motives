from developing.non_symbolic_develop import *

# L**25 + L**13*λ1(X) + L**12*λ1(X) + L**12 + L**11*λ1(X) + L**10*λ1(X)**2 + L**10 + L**9*λ1(X)**2 + L**9*λ1(X) + L**9*λ4(X) + L**8*λ1(X)**2 + 2*L**8*λ1(X)
# + L**8 + L**7*λ1(X)*λ3(X) + L**7*λ4(X) + L**6*λ1(X)**3 + L**6*λ1(X)*λ3(X) + L**6*λ1(X) + L**6*λ2(X) + L**5*λ1(X)**2 + L**5*λ1(X)*λ2(X) + L**5*λ1(X) + L**5*λ4(X)
# + L**4*λ1(X)**2 + L**4*λ1(X)*λ3(X) + L**4*λ2(X) + L**3*λ1(X)**2 + L**3*λ1(X) + L**3 + L**2*λ1(X) + L**2*λ4(X) + L*λ1(X) + 1

g = 2

def get_L_powers(expr):
    terms = sp.Add.make_args(expr)  # works for both single and multi-term
    return list(terms)

def get_terms(expr: LambdaRingExpr) -> tuple[tuple, LambdaRingExpr]:
    lambdas = sorted(expr.free_symbols-{L}, key=lambda x: (len(x.name), x.name))
    if len(lambdas) == 0:
        return (0,), expr
    poly = sp.Poly(expr, *lambdas)
    return {tuple_transform(monom): coef for monom, coef in poly.terms()}

def tuple_transform(v: tuple[int]):
    res = []
    for i in range(len(v)):
        for _ in range(v[i]):
            res.append(i+1)
    res = [0 for _ in range(3-len(res))] + res
    return tuple(res[::-1])

def get_params(monom: tuple[int], coef: LambdaRingExpr):  # por ahora, coef es solo L**k
    candidates = []
    for c1 in range(1, 3*g-2+1):
        for c2 in range(c1+1, 3*g-2+1):
            for c3 in range(c2+1, 3*g-2+1):
                if 15 - c1*monom[0] - c2*monom[1] - c3*monom[2] == coef.args[1]:
                    candidates.append((c1,c2,c3))
    return candidates

def print_sample(sample):
    sample = get_terms(sample)
    print("-------------------------------------")
    for monom in sample.keys():
        print(f"{monom}: {sample[monom]}")
    print("-------------------------------------")

def get_common(motives):
    common = 0
    for monom in motives[0].args:
        if not any(term.could_extract_minus_sign() for m in motives for term in (m-monom).expand().as_ordered_terms()):
            common += monom
    return common

def partial_gl4(X: Curve):
    g = X.g
    top = 3
    res = sum(sym_lambda(X, k1)*sym_lambda(X, k2)*sym_lambda(X, k3)*(L**(k1+2*k2+3*k3)+L**(15*g-15-2*k1-3*k2-4*k3)) for k1 in range(top) for k2 in range(top-k1) for k3 in range(top-k1-k2)) + sym_lambda(X, g-1)**3*L**(6*(g-1))
    res += sym_lambda(X, 0)*sym_lambda(X, 4)*sym_lambda(X, 0)*(L**(0+2*4+3*0)+L**(15*g-15-2*0-3*4-4*0))
    res += sym_lambda(X,3)
    return res

def load_exprs(X: Curve, file_name: str) -> LambdaRingExpr:
    """
    loads expression
    """
    with open(f"{EXPR_DIR}/{file_name}_multiple.pkl", "rb") as file:
        sym_exprs = pickle.load(file)
    exprs = [desymbolize_chow(sym_expr, X).subs(sp.Symbol("L"), L) for sym_expr in sym_exprs]
    return exprs

if __name__ == "__main__":
    # POR HACER: construir formula parcial, restar de obj y buscar con motivefinderclassic
    r = 4
    g = 2
    d = 1
    X = Curve("X", g)
    motives = load_exprs(X, "4_2")
    common = get_common(motives)
    candidates = [motive for motive in motives if get_terms(motive)[(3,0,0)] == L**6 and get_terms(motive)[(4,0,0)] == L**8 + L**3]
    print(motives[0])