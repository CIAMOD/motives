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