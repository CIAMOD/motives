from developing.utils import *
import numpy as np
from functools import reduce



"""
--- VERSION TENSORIAL ---
(a0, a1, ..., ag, k) sera a0*L + a1*λ1(X) + .... + an*λn(X) + k*L, y de ahí saldrán expresiones (sin L)
"""


def outer_prod(*arrays):
    return reduce(lambda a, b: np.tensordot(a, b, axes=0), arrays)


def tensor_power(v, n):
    if n == 0:
        return 1
    args = []
    for i in range(n):
        args += [v, [i]]          
    args += [list(range(n))]     
    return np.einsum(*args)

def pad_tensor(tensor, g):
    dim = g - tensor.ndim
    if dim < 0:
        raise ValueError("negative padding")
    return np.tensordot(tensor_power(np.array([1 if i == 0 else 0 for i in range(g+2)]), dim), tensor, axes=0)


def expr_to_tensor(expr: LambdaRingExpr, X: Curve):
    g = X.g
    expr_sym = symbolize_chow(expr, X)
    lambdas = sorted(expr_sym.free_symbols, key=lambda x: (len(x.name), x.name))
    expr_poly = sp.Poly(expr_sym, *lambdas)
    return expr_poly.terms(), expr_sym


if __name__ == "__main__":
    d = 1
    r = 2
    g = 3
    X = Curve("X", g)
    obj = get_motive_chow(X, r, d)
    tensor = expr_to_tensor(L**4*sym_lambda(X, 3), X)
    x = np.array([0,1,0,0,0])
    y = np.array([1,0,0])
    print(pad_tensor(x, g))