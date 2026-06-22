import sympy as sp

from motives.core.operator.ring_operator import Lambda_

from .power_operations import Wedge, SymPower


def _replace_lambda_by_sym(expr):
    """
    Replace Lambda_ objects by SymPower objects.

    This is only a representation-level change:
        λ2(E)  --->  Sym2(E)

    The algebraic expression is the same.
    """
    return expr.replace(
        lambda x: isinstance(x, Lambda_) and not isinstance(x, SymPower),
        lambda x: SymPower(*x.args),
    )


def to_wedge(self, n: int):
    """
    Exterior power.

    For now we keep wedge powers formal.

    Examples:
        E.to_wedge(2)       -> ∧2(E)
        (E + F).to_wedge(2) -> ∧2(E + F)
        (E * F).to_wedge(2) -> ∧2(E*F)
    """
    if n < 0:
        raise ValueError("n must be nonnegative")

    if n == 0:
        return sp.Integer(1)

    if n == 1:
        return self

    return Wedge(n, self)


def to_sym(self, n: int):
    """
    Symmetric power.

    This one is expanded using the existing motives lambda machinery.

    Internally:
        self.lambda_(n).to_lambda()

    Then we replace printed λ objects by SymPower objects.
    """
    if n < 0:
        raise ValueError("n must be nonnegative")

    expr = sp.expand(self.lambda_(n).to_lambda())
    return _replace_lambda_by_sym(expr)