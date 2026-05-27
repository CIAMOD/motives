from __future__ import annotations
from developing.utils import *
from math import comb

superscripts = str.maketrans("23456789", "²³⁴⁵⁶⁷⁸⁹", "1")


class Poly:
    def __init__(self, data: dict, vars: list[str]) -> None:
        self.data = data  # {(2,1,0): coef, ...}
        self.vars = vars
    
    def __add__(self, other: Poly) -> Poly:
        result = self.data.copy()
        for tup, coef in other.data.items():
            result[tup] = result.get(tup, 0) + coef
            if result[tup] == 0:
                del result[tup]
        return Poly(result, self.vars)
    
    def __sub__(self, other: Poly) -> Poly:
        result = self.data.copy()
        for tup, coef in other.data.items():
            result[tup] = result.get(tup, 0) - coef
            if result[tup] == 0:
                del result[tup]
        return Poly(result, self.vars)
    
    def __mul__(self, other: int | Poly) -> Poly:
        if isinstance(other, int):
            if other == 0:
                return Poly({}, self.vars)
            return Poly({k: v * other for k, v in self.data.items() if v * other != 0}, self.vars)
        result = {}
        for t1, c1 in self.data.items():
            for t2, c2 in other.data.items():
                key = tuple(a + b for a, b in zip(t1, t2))
                result[key] = result.get(key, 0) + c1 * c2
                if result[key] == 0:
                    del result[key]
        return Poly(result, self.vars)
    
    def __repr__(self) -> None:
        expr = " + ".join("".join(([str(self.data[v])] if self.data[v] != 1 or sum(v)==0 else [])  + [f"{self.vars[i]}{str(v[i]).translate(superscripts)}" for i in range(len(v)) if v[i] > 0]) for v in self.data.keys())
        return expr
    
    @classmethod
    def from_sym(cls, expr: sp.Expr) -> Poly:
        lambdas = sorted(expr.free_symbols, key=lambda x: (len(x.name), x.name))
        poly = sp.Poly(expr, lambdas)
        data = {v: coef for v, coef in poly.terms()}
        return cls(data, [s.name for s in lambdas])

class HodgeFinder:
    
    def __init__(self, r: int, g: int, d: int=1) -> None:
        self.r = r
        self.g = g
        self.d = d
        self.curve = Curve("X", g)


def draw_diamond(poly, deg=-1, val_size=4, indx_len=2):
        if len(poly.vars) != 2:
            raise ValueError("Poly must be of degree 2")
        if deg == -1:
            deg = max([sum(v) for v in poly.data.keys()])
        for d in range(deg+1):
            print((indx_len-len(str(d)))*" " + str(d) + "     ", end="")
            print(" "*int(val_size*(deg-d)/2), end="")
            for i in range(d+1):
                j = d - i
                coef_str = str(poly.data[(i,j)] if (i,j) in poly.data.keys() else 0)
                print(coef_str + " "*(val_size-len(coef_str)), end="")
            print("")
        for d in range(deg+1, 2*deg+1):
            print((indx_len-len(str(d)))*" " + str(d) + "     ", end="")
            print(" "*int(val_size*(d-deg)/2), end="")
            for i in range(deg+1):
                j = d - i
                if j <= deg:
                    coef_str = str(poly.data[(i,j)] if (i,j) in poly.data.keys() else 0)
                    print(coef_str + " "*(val_size-len(coef_str)), end="")
            print("")


def lambda_chow2hodge(g: int, k: int) -> sp.Expr:
    # computes hodge polynomial of lambdak(h1(X)), where g is the genus of X
    return sum(comb(g,i)*comb(g,k-i)*sp.Symbol("X")**i*sp.Symbol("Y")**(k-i) for i in range(k+1))

def chow2hodge(motive: LambdaRingExpr, curve: Curve) -> Poly:
    # computes hodge polynomial of motive
    chow = curve.curve_chow
    X, Y = sp.Symbol("X"), sp.Symbol("Y")
    subs = {
    L: X*Y,  
    **{symbolize_chow(chow.get_lambda_var(i), curve): lambda_chow2hodge(curve.g, i) for i in range(1, curve.g+1)}   # mas feo que una nevera por detrás
    }
    return Poly.from_sym(motive.subs(subs).expand())

if __name__ == "__main__":
    r = 4
    g = 2
    X = Curve("X", g=g)
    m = symbolize_chow(get_motive_chow(X, r=r, d=1), X)   #tiene q estar simbolizado importante
    print(m)
    m = ((m-1-L**15)/L).expand()
    m_hodge = chow2hodge(m, X)
    draw_diamond(m_hodge, deg=(r**2-1)*(g-1)-2)