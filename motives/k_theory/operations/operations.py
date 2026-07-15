import sympy as sp

from motives.core.operator.ring_operator import Lambda_, Sigma

from .power_operations import Wedge, SymPower


def wedge(self, n: int):
    """
    Formal exterior power.

    This only creates the formal Wedge object:

        self.wedge(n)  ->  ∧n(self)

    It does not expand the expression. Use to_wedge(n) for the expanded version.
    """
    if n < 0:
        raise ValueError("n must be nonnegative")
    if n == 0:
        return sp.Integer(1)
    if n == 1:
        return self

    return Wedge(n, self)


def to_wedge(self, n: int | None = None):
    """
    Expanded exterior power.

    If n is given, this computes and expands the n-th exterior power of self:

        self.to_wedge(n)

    If n is not given and self is already a Wedge object, this expands that
    formal Wedge object:

        self.wedge(n).to_wedge()
    """
    if n is None:
        if not isinstance(self, Wedge):
            return self

        expr = sp.expand(self.to_sigma())
        return _replace_sigma_by_wedge(expr)

    if n < 0:
        raise ValueError("n must be nonnegative")
    if n == 0:
        return sp.Integer(1)
    if n == 1:
        return self

    expr = sp.expand(self.sigma(n).to_sigma())
    return _replace_sigma_by_wedge(expr)


def _replace_sigma_by_wedge(expr):
    """
    Replace Sigma objects by Wedge objects.

    This is only a representation-level change:
        σ2(E)  --->  ∧2(E)

    The algebraic expression is the same.
    """
    return expr.replace(
        lambda x: isinstance(x, Sigma) and not isinstance(x, Wedge),
        lambda x: Wedge(*x.args)
    )


def sym(self, n: int):
    """
    Formal symmetric power.

    This only creates the formal SymPower object:

        self.sym(n)  ->  Symn(self)

    It does not expand the expression. Use to_sym(n) for the expanded version.
    """
    if n < 0:
        raise ValueError("n must be nonnegative")
    if n == 0:
        return sp.Integer(1)
    if n == 1:
        return self

    return SymPower(n, self)


def to_sym(self, n: int | None = None):
    """
    Expanded symmetric power.

    If n is given, this computes and expands the n-th symmetric power of self:

        self.to_sym(n)

    If n is not given and self is already a SymPower object, this expands that
    formal SymPower object:

        self.sym(n).to_sym()
    """
    if n is None:
        if not isinstance(self, SymPower):
            return self

        expr = sp.expand(self.to_lambda())
        return _replace_lambda_by_sym(expr)

    if n < 0:
        raise ValueError("n must be nonnegative")
    if n == 0:
        return sp.Integer(1)
    if n == 1:
        return self

    expr = sp.expand(self.lambda_(n).to_lambda())
    return _replace_lambda_by_sym(expr)


def _replace_lambda_by_sym(expr):
    """
    Replace Lambda_ objects by SymPower objects.

    This is only a representation-level change:
        λ2(E)  --->  Sym2(E)

    The algebraic expression is the same.
    """
    return expr.replace(
        lambda x: isinstance(x, Lambda_) and not isinstance(x, SymPower),
        lambda x: SymPower(*x.args)
    )