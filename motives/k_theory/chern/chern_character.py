from __future__ import annotations

from typing import Optional

import sympy as sp

from ..objects.vector_bundle import VectorBundle
from ..objects.dual_bundle import DualBundle
from ..objects.determinant_bundle import DeterminantBundle
from ..objects.power_bundles import SymPower, Wedge


def _component(ch: tuple[sp.Expr, ...], i: int) -> sp.Expr:
    """
    Return a homogeneous component of a truncated Chern character.

    Parameters
    ----------
    ch : tuple of sympy.Expr
        Chern-character components indexed by degree.
    i : int
        Requested degree.

    Returns
    -------
    sympy.Expr
        ``ch[i]`` when the degree is stored, and zero otherwise.

    Notes
    -----
    Returning zero outside the stored range treats the tuple as a truncated
    graded series and allows tuples with different truncation degrees to be
    combined safely.
    """
    if 0 <= i < len(ch):
        return sp.sympify(ch[i])
    return sp.Integer(0)


def _zero_ch(max_chern_degree: int) -> tuple[sp.Expr, ...]:
    """
    Return the Chern character of the additive zero object.

    Parameters
    ----------
    max_chern_degree : int
        Highest degree to include.

    Returns
    -------
    tuple of sympy.Expr
        Tuple ``(0, 0, ..., 0)`` containing
        ``max_chern_degree + 1`` components.
    """
    return tuple(sp.Integer(0) for _ in range(max_chern_degree + 1))


def _one_ch(max_chern_degree: int) -> tuple[sp.Expr, ...]:
    """
    Return the multiplicative identity Chern character.

    Parameters
    ----------
    max_chern_degree : int
        Highest degree to include.

    Returns
    -------
    tuple of sympy.Expr
        Tuple ``(1, 0, ..., 0)``.

    Notes
    -----
    This is the Chern character of the trivial line bundle and is the identity
    for the graded product used to model tensor products.
    """
    return tuple(
        [sp.Integer(1)]
        + [sp.Integer(0) for _ in range(max_chern_degree)]
    )


def _ch_add(A: tuple[sp.Expr, ...], B: tuple[sp.Expr, ...], max_chern_degree: int) -> tuple[sp.Expr, ...]:
    """
    Add two Chern characters component by component.

    This implements additivity under direct sums:

    ``ch(E ⊕ F) = ch(E) + ch(F)``.

    Equivalently, for every degree ``k``,

    ``ch_k(E ⊕ F) = ch_k(E) + ch_k(F)``.

    Parameters
    ----------
    A, B : tuple of sympy.Expr
        Chern-character components of the two operands.
    max_chern_degree : int
        Highest degree retained in the result.

    Returns
    -------
    tuple of sympy.Expr
        Componentwise sum truncated through ``max_chern_degree``.
    """
    return tuple(
        _component(A, i) + _component(B, i)
        for i in range(max_chern_degree + 1)
    )


def _ch_scalar_mul(A: tuple[sp.Expr, ...], scalar: sp.Expr, max_chern_degree: int) -> tuple[sp.Expr, ...]:
    """
    Multiply every Chern-character component by a scalar.

    For a scalar ``a`` and a K-theory class ``E``, this implements

    ``ch(aE) = a · ch(E)``.

    Parameters
    ----------
    A : tuple of sympy.Expr
        Input Chern-character components.
    scalar : sympy.Expr
        Scalar multiplier.
    max_chern_degree : int
        Highest degree retained in the result.

    Returns
    -------
    tuple of sympy.Expr
        Scaled Chern-character components.
    """
    scalar = sp.sympify(scalar)
    return tuple(
        scalar * _component(A, i)
        for i in range(max_chern_degree + 1)
    )


def _ch_determinant(A: tuple[sp.Expr, ...], max_chern_degree: int) -> tuple[sp.Expr, ...]:
    """Compute the Chern character of a determinant line bundle.

    If ``A`` represents ``ch(E)``, then the determinant satisfies

        chi(det(E)) = ch1(E)^i / i!.

    Parameters
    ----------
    A : tuple of sympy.Expr
        Chern-character components of the original bundle.
    max_chern_degree : int
        Highest homogeneous degree retained.

    Returns
    -------
    tuple of sympy.Expr
        Chern-character components of ``det(E)``.
    """
    ch1 = _component(A, 1)
    return tuple(sp.Integer(1) if i == 0 else sp.expand(ch1**i / sp.factorial(i)) for i in range(max_chern_degree + 1))


def _ch_dual(A: tuple[sp.Expr, ...], max_chern_degree: int) -> tuple[sp.Expr, ...]:
    """
    Compute the Chern character of a dual vector-bundle expression.

    If ``A`` represents ``ch(E)``, the returned tuple represents
    ``ch(E^∨)``. Dualization changes every Chern root ``x_j`` into ``-x_j``
    and therefore acts on each homogeneous component according to

    ``ch_i(E^∨) = (-1)^i ch_i(E)``.

    Thus, even-degree components remain unchanged and odd-degree components
    change sign.

    Parameters
    ----------
    A
        Tuple representing ``ch(E)``.
    max_chern_degree : int
        Maximum homogeneous degree to compute.

    Returns
    -------
    tuple[sympy.Expr, ...]
        Tuple representing ``ch(E^∨)``.
    """
    return tuple(sp.Integer(-1) ** i * _component(A, i) for i in range(max_chern_degree + 1))


def _ch_mul(A: tuple[sp.Expr, ...], B: tuple[sp.Expr, ...], max_chern_degree: int) -> tuple[sp.Expr, ...]:
    """
    Multiply two Chern characters as truncated graded series.

    This implements multiplicativity under tensor products:

    ``ch(E ⊗ F) = ch(E) · ch(F)``.

    If ``A_i = ch_i(E)`` and ``B_j = ch_j(F)``, the homogeneous component of
    degree ``k`` is

    ``ch_k(E ⊗ F) = Σ_{i=0}^k A_i B_{k-i}``.

    Parameters
    ----------
    A, B : tuple of sympy.Expr
        Chern-character components of the operands.
    max_chern_degree : int
        Highest total degree retained.

    Returns
    -------
    tuple of sympy.Expr
        Graded convolution product truncated through
        ``max_chern_degree``.
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
    Apply the k-th Adams operation to Chern-character components.

    If ``A_i = ch_i(E)``, the Adams operation satisfies

    ``ch_i(ψᵏ(E)) = kⁱ ch_i(E)``.

    Therefore,

    ``ψᵏ(A) = (A_0, k A_1, k² A_2, ..., kᴺ A_N)``.

    Parameters
    ----------
    A : tuple of sympy.Expr
        Chern-character components.
    k : int
        Positive Adams-operation index.
    max_chern_degree : int
        Highest degree retained.

    Returns
    -------
    tuple of sympy.Expr
        Components of the Adams-transformed Chern character.

    Raises
    ------
    ValueError
        If ``k < 1``.
    """
    if k < 1:
        raise ValueError("Adams operations require k >= 1.")

    return tuple(
        sp.Integer(k) ** i * _component(A, i)
        for i in range(max_chern_degree + 1)
    )


def _ch_sym_power(A: tuple[sp.Expr, ...], n: int, max_chern_degree: int) -> tuple[sp.Expr, ...]:
    """
    Compute the Chern character of an n-th symmetric power.

    Let

    ``h_m = ch(Symᵐ(E))``

    and ``h_0 = 1``. The complete-symmetric-function generating series is

    ``Σ_{m>=0} h_m t^m = exp(Σ_{i>=1} ψⁱ(ch(E)) tⁱ / i)``.

    Differentiating this identity gives the recurrence implemented here:

    ``m h_m = Σ_{i=1}^m ψⁱ(ch(E)) · h_{m-i}``.

    Thus,

    ``h_m = (1/m) Σ_{i=1}^m ψⁱ(ch(E)) · h_{m-i}``.

    Products in this formula are graded Chern-character products and are
    truncated through ``max_chern_degree``.

    Parameters
    ----------
    A : tuple of sympy.Expr
        Chern character ``ch(E)``.
    n : int
        Symmetric-power degree.
    max_chern_degree : int
        Highest homogeneous Chern degree retained.

    Returns
    -------
    tuple of sympy.Expr
        Truncated Chern character ``ch(Symⁿ(E))``.

    Raises
    ------
    ValueError
        If ``n`` is negative.
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
    Compute the Chern character of an n-th exterior power.

    Let

    ``e_m = ch(Λᵐ(E))``

    and ``e_0 = 1``. The elementary-symmetric-function generating series is

    ``Σ_{m>=0} e_m t^m =
    exp(Σ_{i>=1} (-1)^(i-1) ψⁱ(ch(E)) tⁱ / i)``.

    This gives the Newton recurrence

    ``m e_m =
    Σ_{i=1}^m (-1)^(i-1) ψⁱ(ch(E)) · e_{m-i}``.

    Therefore,

    ``e_m =
    (1/m) Σ_{i=1}^m (-1)^(i-1) ψⁱ(ch(E)) · e_{m-i}``.

    Parameters
    ----------
    A : tuple of sympy.Expr
        Chern character ``ch(E)``.
    n : int
        Exterior-power degree.
    max_chern_degree : int
        Highest homogeneous Chern degree retained.

    Returns
    -------
    tuple of sympy.Expr
        Truncated Chern character ``ch(Λⁿ(E))``.

    Raises
    ------
    ValueError
        If ``n`` is negative.
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
    """Return the base vector bundles contained in an expression.

    Derived bundles such as ``DualBundle`` and ``DeterminantBundle`` are
    recursively replaced by the original bundles from which their
    characteristic data are computed.

    Parameters
    ----------
    expr : sympy.Expr
        Expression to inspect.

    Returns
    -------
    tuple of VectorBundle
        Non-repeated base vector bundles contained in the expression.
    """
    if isinstance(expr, (DualBundle, DeterminantBundle)):
        return _vector_bundles(expr.bundle)

    if isinstance(expr, VectorBundle):
        return (expr,)

    try:
        atoms = expr.atoms(VectorBundle)
    except AttributeError:
        return ()

    bundles = []

    for atom in atoms:
        for bundle in _vector_bundles(atom):
            if bundle not in bundles:
                bundles.append(bundle)

    return tuple(bundles)


def _validate_same_scheme(expr: sp.Expr) -> None:
    """
    Validate that all explicit bundles share the same base scheme.

    Parameters
    ----------
    expr : sympy.Expr
        Expression containing vector bundles.

    Raises
    ------
    ValueError
        If two vector bundles with explicit, non-``None`` schemes are defined
        over different schemes.

    Notes
    -----
    Bundles whose scheme is ``None`` are treated as purely formal symbols and
    do not participate in the compatibility check.
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
    Determine the maximum available Chern degree for a bundle.

    The stored ``bundle.max_degree`` is preferred. If it is unavailable, an
    integer dimension of the base scheme is used. If neither source provides
    an integer degree, the fallback value five is returned.

    Parameters
    ----------
    bundle : VectorBundle
        Bundle whose available degree is requested.

    Returns
    -------
    int
        Maximum usable Chern degree.
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
    Infer a common Chern-character truncation degree for an expression.

    Parameters
    ----------
    expr : sympy.Expr
        Expression containing zero or more vector bundles.

    Returns
    -------
    int
        Minimum available maximum degree among the bundles in ``expr``.
        Returns five if no vector bundle is present.

    Notes
    -----
    The minimum is used so that every bundle occurring in the computation has
    data available through the selected degree.
    """
    bundles = _vector_bundles(expr)

    if not bundles:
        return 5

    return min(_bundle_max_chern_degree(bundle) for bundle in bundles)


def _is_scalar_expression(expr: sp.Expr) -> bool:
    """
    Return whether an expression contains no vector-bundle atoms.

    Scalar expressions are interpreted as scalar multiples of the
    multiplicative unit. Consequently, a scalar ``a`` has Chern character

    ``ch(a) = (a, 0, ..., 0)``.

    Parameters
    ----------
    expr : sympy.Expr
        Expression to inspect.

    Returns
    -------
    bool
        ``True`` when no ``VectorBundle`` occurs in the expression.
    """
    return len(_vector_bundles(expr)) == 0


def _compute_chern_character(expr: sp.Expr, max_chern_degree: int) -> tuple[sp.Expr, ...]:
    """
    Recursively evaluate the Chern character of an expression tree.

    The tree is processed from its leaves to its root according to the
    following K-theoretic interpretation:

    - ``VectorBundle``: use its stored Chern-character components.
    - ``Dual``: compute the character of the operand and apply ``ch_i(E^∨) = (-1)^i ch_i(E)``.
    - ``Add``: interpret addition as direct sum and add characters.
    - ``Mul``: interpret multiplication as tensor product and multiply the
    characters as graded series.
    - ``Pow``: interpret a non-negative integer power as repeated tensor
    product.
    - ``SymPower``: use the Adams-operation recurrence for symmetric powers.
    - ``Wedge``: use the Adams-operation recurrence for exterior powers.
    - Scalar expression: place the scalar in degree zero.

    Parameters
    ----------
    expr : sympy.Expr
        Expression to evaluate.
    max_chern_degree : int
        Highest homogeneous degree retained.

    Returns
    -------
    tuple of sympy.Expr
        Chern-character components through ``max_chern_degree``.

    Raises
    ------
    NotImplementedError
        If a power has a non-integer or negative exponent, or if the expression
        contains an unsupported node type.
    """
    if isinstance(expr, DualBundle):
        inner_ch = _compute_chern_character(expr.bundle, max_chern_degree)
        return _ch_dual(inner_ch, max_chern_degree)

    if isinstance(expr, DeterminantBundle):
        inner_ch = _compute_chern_character(expr.bundle, max_chern_degree)
        return _ch_determinant(inner_ch, max_chern_degree)

    if isinstance(expr, VectorBundle):
        stored = expr.chern_character
        return tuple(_component(stored, i) for i in range(max_chern_degree + 1))

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


def chern_character(expr: sp.Expr, component: Optional[int] = None, *, max_chern_degree: Optional[int] = None) -> tuple[sp.Expr, ...] | sp.Expr:
    """
    Compute the Chern character of an expression of vector bundles.

    If ``component`` is ``None``, return the full tuple
    ``(ch_0, ch_1, ..., ch_max_chern_degree)``. If ``component`` is provided,
    return only that homogeneous component.

    The expression is interpreted using the K-theoretic rules:

    ``E + F`` -> direct sum

    ``E * F`` -> tensor product

    ``E ** n`` -> repeated tensor product

    ``Dual(E)`` -> dual vector bundle

    ``Hom(E, F)`` -> ``E^∨ ⊗ F``

    ``End(E)`` -> ``E^∨ ⊗ E``

    ``Det(E)`` -> ``Λ^rank(E)(E)``

    ``SymPower(E, n)`` -> n-th symmetric power

    ``Wedge(E, n)`` -> n-th exterior power

    Parameters
    ----------
    expr : sympy.Expr
        Expression built from vector bundles using sums, tensor products,
        powers, duals, homomorphism bundles, endomorphism bundles,
        determinants, symmetric powers, and exterior powers.
    component : int, optional
        Homogeneous degree to return. If omitted, the full tuple is returned.
    max_chern_degree : int, optional
        Highest homogeneous degree to compute. If omitted, it is inferred from
        the bundles in ``expr``. When ``component`` is supplied and no maximum
        degree is given, computation stops at that component.

    Returns
    -------
    tuple of sympy.Expr or sympy.Expr
        Full tuple

        ``(ch_0(expr), ch_1(expr), ..., ch_N(expr))``

        or the selected homogeneous component.

    Raises
    ------
    TypeError
        If ``component`` is not an integer.
    ValueError
        If ``component`` or ``max_chern_degree`` is negative, if
        ``max_chern_degree < component``, or if the expression contains
        bundles over incompatible schemes.
    NotImplementedError
        If the expression contains an unsupported operation or a power with a
        negative or non-integer exponent.

    Notes
    -----
    The degree-zero component is the virtual rank:

    ``ch_0(E) = rank(E)``.

    All results are truncated after ``max_chern_degree``.

    Examples
    --------
    Compute all available components:

    ``chern_character(E * F)``

    Compute through degree three:

    ``chern_character(E * F, max_chern_degree=3)``

    Compute only degree two:

    ``chern_character(E * F, component=2)``
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