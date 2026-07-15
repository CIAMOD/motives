from __future__ import annotations

from typing import Optional

import sympy as sp

from ..objects.vector_bundle import VectorBundle
from ..operations.power_operations import SymPower, Wedge


def _component(ch: tuple[sp.Expr, ...], i: int) -> sp.Expr:
    """
    Return the i-th homogeneous component of a Chern character.

    If the requested component is outside the stored range, return 0. This lets
    computations safely combine Chern characters stored up to different degrees.
    """
    if 0 <= i < len(ch):
        return sp.sympify(ch[i])
    return sp.Integer(0)


def _zero_ch(max_chern_degree: int) -> tuple[sp.Expr, ...]:
    """
    Return the Chern character of the zero object.

    The result is the tuple (0, ..., 0) with components from degree 0 up to
    max_chern_degree.
    """
    return tuple(sp.Integer(0) for _ in range(max_chern_degree + 1))


def _one_ch(max_chern_degree: int) -> tuple[sp.Expr, ...]:
    """
    Return the Chern character of the trivial line bundle.

    The result is (1, 0, ..., 0), which is the multiplicative identity for
    tensor-product computations.
    """
    return tuple(
        [sp.Integer(1)]
        + [sp.Integer(0) for _ in range(max_chern_degree)]
    )


def _ch_add(A: tuple[sp.Expr, ...], B: tuple[sp.Expr, ...], max_chern_degree: int) -> tuple[sp.Expr, ...]:
    """
    Add two Chern characters componentwise.

    This implements the identity ch(E + F) = ch(E) + ch(F), where + denotes
    direct sum in K-theory.
    """
    return tuple(
        _component(A, i) + _component(B, i)
        for i in range(max_chern_degree + 1)
    )


def _ch_scalar_mul(A: tuple[sp.Expr, ...], scalar: sp.Expr, max_chern_degree: int) -> tuple[sp.Expr, ...]:
    """
    Multiply every component of a Chern character by a scalar.

    This is used for scalar multiples such as nE and for rational coefficients
    appearing in the symmetric and exterior power recurrences.
    """
    scalar = sp.sympify(scalar)
    return tuple(
        scalar * _component(A, i)
        for i in range(max_chern_degree + 1)
    )


def _ch_mul(A: tuple[sp.Expr, ...], B: tuple[sp.Expr, ...], max_chern_degree: int) -> tuple[sp.Expr, ...]:
    """
    Multiply two Chern characters as truncated graded series.

    This implements ch(E * F) = ch(E) ch(F), where * denotes tensor product.
    The k-th component is the convolution sum

        sum(A_i * B_{k-i} for i = 0, ..., k).
    """
    return tuple(
        sum(
            (
                _component(A, i) * _component(B, k - i)
                for i in range(k + 1)
            ),
            sp.Integer(0),
        )
        for k in range(max_chern_degree + 1)
    )


def _adams_chern(A: tuple[sp.Expr, ...], k: int, max_chern_degree: int) -> tuple[sp.Expr, ...]:
    """
    Apply the k-th Adams operation to a Chern character.

    On Chern character components, the Adams operation acts by

        psi^k(ch_i) = k^i ch_i.

    Thus degree 0 is unchanged, degree 1 is multiplied by k, degree 2 by k^2,
    and so on.
    """
    if k < 1:
        raise ValueError("Adams operations require k >= 1.")

    return tuple(
        sp.Integer(k) ** i * _component(A, i)
        for i in range(max_chern_degree + 1)
    )


def _ch_sym_power(A: tuple[sp.Expr, ...], n: int, max_chern_degree: int) -> tuple[sp.Expr, ...]:
    """
    Compute the Chern character of the n-th symmetric power.

    The input A represents ch(E). The function returns ch(Sym^n(E)) using the
    recurrence for complete symmetric functions:

        m h_m = sum(psi^i(E) h_{m-i}, i = 1, ..., m),

    with h_0 = 1 and h_m = ch(Sym^m(E)).
    """
    if n < 0:
        raise ValueError("Symmetric powers require n >= 0.")

    h: list[tuple[sp.Expr, ...] | None] = [None] * (n + 1)
    h[0] = _one_ch(max_chern_degree)

    for m in range(1, n + 1):
        acc = _zero_ch(max_chern_degree)

        for i in range(1, m + 1):
            psi_i = _adams_chern(A, i, max_chern_degree)
            term = _ch_mul(psi_i, h[m - i], max_chern_degree)
            acc = _ch_add(acc, term, max_chern_degree)

        h[m] = _ch_scalar_mul(acc, sp.Rational(1, m), max_chern_degree)

    return h[n]


def _ch_wedge_power(A: tuple[sp.Expr, ...], n: int, max_chern_degree: int) -> tuple[sp.Expr, ...]:
    """
    Compute the Chern character of the n-th exterior power.

    The input A represents ch(E). The function returns ch(Λ^n(E)) using the
    recurrence for elementary symmetric functions:

        m e_m = sum((-1)^(i-1) psi^i(E) e_{m-i}, i = 1, ..., m),

    with e_0 = 1 and e_m = ch(Λ^m(E)).
    """
    if n < 0:
        raise ValueError("Exterior powers require n >= 0.")

    e: list[tuple[sp.Expr, ...] | None] = [None] * (n + 1)
    e[0] = _one_ch(max_chern_degree)

    for m in range(1, n + 1):
        acc = _zero_ch(max_chern_degree)

        for i in range(1, m + 1):
            sign = sp.Integer((-1) ** (i - 1))
            psi_i = _adams_chern(A, i, max_chern_degree)
            term = _ch_mul(psi_i, e[m - i], max_chern_degree)
            acc = _ch_add(
                acc,
                _ch_scalar_mul(term, sign, max_chern_degree),
                max_chern_degree,
            )

        e[m] = _ch_scalar_mul(acc, sp.Rational(1, m), max_chern_degree)

    return e[n]


def _vector_bundles(expr: sp.Expr) -> tuple[VectorBundle, ...]:
    """
    Return all VectorBundle atoms appearing in an expression.

    If expr itself is a VectorBundle, return it directly. Otherwise, use SymPy's
    atom search. Non-SymPy objects return an empty tuple.
    """
    if isinstance(expr, VectorBundle):
        return (expr,)

    try:
        return tuple(expr.atoms(VectorBundle))
    except AttributeError:
        return ()


def _validate_same_scheme(expr: sp.Expr) -> None:
    """
    Validate that all explicit vector bundles live over the same scheme.

    Bundles with scheme=None are treated as formal symbols and ignored. If two
    explicit, non-None schemes differ, the expression is considered geometrically
    invalid and a ValueError is raised.
    """
    bundles = _vector_bundles(expr)
    schemes = [
        bundle.scheme
        for bundle in bundles
        if getattr(bundle, "scheme", None) is not None
    ]

    if not schemes:
        return

    first = schemes[0]

    for scheme in schemes[1:]:
        if scheme != first:
            raise ValueError(
                "All vector bundles in the expression must be defined over "
                "the same scheme."
            )


def _bundle_max_chern_degree(bundle: VectorBundle) -> int:
    """
    Return the maximum Chern character degree available for a bundle.

    Prefer the bundle's stored max_degree when available. Otherwise, use the
    dimension of the base scheme if it is an integer. If neither is available,
    fall back to 5.
    """
    if hasattr(bundle, "max_degree"):
        return int(bundle.max_degree)

    scheme = getattr(bundle, "scheme", None)
    dimension = getattr(scheme, "dimension", None)

    if isinstance(dimension, int):
        return dimension

    return 5


def _infer_max_chern_degree(expr: sp.Expr) -> int:
    """
    Infer a truncation degree from the vector bundles in an expression.

    The returned value is the minimum available maximum Chern degree among all
    vector bundles appearing in expr. If no vector bundle is found, return 5.
    """
    bundles = _vector_bundles(expr)

    if not bundles:
        return 5

    return min(_bundle_max_chern_degree(bundle) for bundle in bundles)


def _is_scalar_expression(expr: sp.Expr) -> bool:
    """
    Return whether an expression contains no vector bundles.

    Such expressions are interpreted as scalar multiples of the trivial bundle,
    so their Chern character is concentrated in degree 0.
    """
    return len(_vector_bundles(expr)) == 0


def _compute_chern_character(expr: sp.Expr, max_chern_degree: int) -> tuple[sp.Expr, ...]:
    """
    Recursively compute the Chern character of an expression.

    The expression tree is evaluated bottom-up. VectorBundle nodes provide their
    stored Chern character. Add nodes are direct sums, Mul nodes are tensor
    products, Pow nodes are repeated tensor products, and SymPower/Wedge nodes
    are computed using Adams-operation recurrences.
    """
    if isinstance(expr, VectorBundle):
        stored = expr.chern_character
        return tuple(
            _component(stored, i)
            for i in range(max_chern_degree + 1)
        )

    if isinstance(expr, SymPower):
        inner_expr = expr.child
        n = int(expr.degree)
        inner_ch = _compute_chern_character(inner_expr, max_chern_degree)
        return _ch_sym_power(inner_ch, n, max_chern_degree)

    if isinstance(expr, Wedge):
        inner_expr = expr.child
        n = int(expr.degree)
        inner_ch = _compute_chern_character(inner_expr, max_chern_degree)
        return _ch_wedge_power(inner_ch, n, max_chern_degree)

    if isinstance(expr, sp.Add):
        result = _zero_ch(max_chern_degree)

        for arg in expr.args:
            arg_ch = _compute_chern_character(arg, max_chern_degree)
            result = _ch_add(result, arg_ch, max_chern_degree)

        return result

    if isinstance(expr, sp.Mul):
        result = _one_ch(max_chern_degree)

        for arg in expr.args:
            arg_ch = _compute_chern_character(arg, max_chern_degree)
            result = _ch_mul(result, arg_ch, max_chern_degree)

        return result

    if isinstance(expr, sp.Pow):
        exp = expr.exp

        if not isinstance(exp, (int, sp.Integer)):
            raise NotImplementedError(
                "Chern character computation only supports integer powers."
            )

        exp = int(exp)

        if exp < 0:
            raise NotImplementedError(
                "Chern character computation only supports non-negative powers."
            )

        base_ch = _compute_chern_character(expr.base, max_chern_degree)
        result = _one_ch(max_chern_degree)

        for _ in range(exp):
            result = _ch_mul(result, base_ch, max_chern_degree)

        return result

    if _is_scalar_expression(expr):
        return tuple(
            [sp.sympify(expr)]
            + [sp.Integer(0) for _ in range(max_chern_degree)]
        )

    raise NotImplementedError(
        f"Chern character computation is not supported for expressions "
        f"of type {type(expr)}."
    )


def chern_character(
    expr: sp.Expr,
    component: Optional[int] = None,
    *,
    max_chern_degree: Optional[int] = None,
) -> tuple[sp.Expr, ...] | sp.Expr:
    """
    Compute the Chern character of an expression of vector bundles.

    If component is None, return the full tuple

        (ch_0, ch_1, ..., ch_max_chern_degree).

    If component is provided, return only that homogeneous component. The
    expression is interpreted using the K-theoretic rules:

        E + F          -> direct sum
        E * F          -> tensor product
        E ** n         -> repeated tensor product
        SymPower(E, n) -> n-th symmetric power
        Wedge(E, n)    -> n-th exterior power

    All vector bundles with explicit schemes must be defined over the same
    scheme. Bundles with scheme=None are treated formally.

    Examples
    --------
    chern_character(E * F)
    chern_character(E * F, max_chern_degree=3)
    chern_character(E * F, component=2)
    """
    expr = sp.sympify(expr)

    _validate_same_scheme(expr)

    if component is not None:
        if not isinstance(component, (int, sp.Integer)):
            raise TypeError("component must be an integer.")

        component = int(component)

        if component < 0:
            raise ValueError("component must be non-negative.")

        if max_chern_degree is None:
            max_chern_degree = component

        if max_chern_degree < component:
            raise ValueError("max_chern_degree must be >= component.")

    if max_chern_degree is None:
        max_chern_degree = _infer_max_chern_degree(expr)

    if max_chern_degree < 0:
        raise ValueError("max_chern_degree must be non-negative.")

    full_ch = _compute_chern_character(expr, max_chern_degree)

    if component is not None:
        return _component(full_ch, component)

    return full_ch