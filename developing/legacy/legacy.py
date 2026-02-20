def find_motive_X(obj: sp.Expr, X: Curve, H: CurveChow, coefficients: list[sp.Expr], max_degree: int=-1):
    """
    return positive polynomial in X equal to motive (aquí la chicha), obj dado en suma positiva en h1
    """
    if max_degree == -1:
        max_degree = 2 * X.g
    if max_degree == 0:
        return obj
    
    print(max_degree)
 
    rest = obj
    big_part = 0
    for i in range(max_degree, -1, -1):
        x_power_h = X.get_lambda_var(i)*X.get_lambda_var(max_degree)
        x_power_sym = sym_lambda(X, i)*sym_lambda(X, max_degree)
        total_coef = 0
        for coef in coefficients:   
            candidate = (rest - coef*x_power_h).expand().simplify()
            while not any(term.could_extract_minus_sign() for term in candidate.as_ordered_terms()):
                total_coef += coef 
                candidate = (candidate - coef*x_power_h).expand().simplify()
        big_part += total_coef*x_power_sym
        rest -= total_coef*x_power_h

    print(big_part)
    motive = big_part + find_motive_X(rest, X, H, coefficients, max_degree=max_degree-1)
    return motive.expand().simplify()


def find_motive_bfs(obj: LambdaRingExpr, X: Curve, coefs: list[LambdaRingExpr], monoms_H: list[LambdaRingExpr], monoms_X: list[LambdaRingExpr], n_rounds: int) -> list[LambdaRingExpr]: 
    """
    compute all possible motivic decompositions given monomials (high degree) and coefficients
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


def get_full_monoms(monoms, coefs, max_dim):
    min_degree = min(monoms.keys())
    max_degree = max(monoms.keys())
    full_monoms = {k: [] for k in range(min_degree, max_dim+1)}
    for degree in range(max_degree, min_degree-1, -1):
        for k in range(max_dim-degree+1):
            full_monoms[degree+k] += [(coefs[k]*monom_X, coefs[k]*monom_H) for monom_X, monom_H in monoms[degree]]
    return full_monoms

def find_motive_bfs_new(obj: LambdaRingExpr, X: Curve, coefs: list[LambdaRingExpr], monoms, n_rounds: int, max_dim: int=-1) -> list[LambdaRingExpr]: 
    """
    compute all possible motivic decompositions given monomials (high degree) and coefficients
    """
    full_monoms = get_full_monoms(monoms, coefs, max_dim)
    candidates = {(obj, 0)}
    for degree in sorted(full_monoms.keys(), reverse=True):
        for monom_X, monom_H in full_monoms[degree]:
            for _ in range(n_rounds):
                new_candidates = set()
                for candidate, big_part in tqdm(candidates):
                    m = (candidate - monom_H).expand()
                    if monom_H == 0:
                        print(m)
                    if not any(term.could_extract_minus_sign() for term in m.as_ordered_terms()):
                        new_candidates.add((m, big_part+monom_X))
                candidates = candidates.union(new_candidates)
    motives = {(big_part + m).expand() for candidate, big_part in tqdm(candidates) if (m := find_motive_low(candidate, X)) is not None}
    return motives