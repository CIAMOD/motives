from __future__ import annotations

from typing import Optional

import sympy as sp

from ..objects.vector_bundle import VectorBundle
from .chern_character import (
    chern_character,
    _component,
    _compute_chern_character,
    _infer_max_chern_degree,
    _validate_same_scheme,
)


def _validate_component_and_degree(expr: sp.Expr, component: Optional[int], max_chern_degree: Optional[int]) -> tuple[Optional[int], int]:
    """
    Validate and normalize the requested component and truncation degree.

    Parameters
    ----------
    expr : sympy.Expr
        Expression whose natural maximum degree may need to be inferred.
    component : int or None
        Requested Chern-class degree.
    max_chern_degree : int or None
        Explicit truncation degree.

    Returns
    -------
    tuple
        Normalized pair ``(component, max_chern_degree)``.

    Raises
    ------
    TypeError
        If ``component`` is not an integer.
    ValueError
        If either degree is negative or if
        ``max_chern_degree < component``.

    Notes
    -----
    If a component is requested without an explicit maximum degree, the
    computation is truncated exactly at that component. If neither value is
    given, the degree is inferred from the vector bundles in ``expr``.
    """
    if component is not None:
        if not isinstance(component, (int, sp.Integer)):
            raise TypeError("component must be an integer.")

        component = int(component)

        if component < 0:
            raise ValueError("component must be non-negative.")

        if max_chern_degree is None:
            max_chern_degree = component

    if max_chern_degree is None:
        max_chern_degree = _infer_max_chern_degree(expr)

    max_chern_degree = int(max_chern_degree)

    if max_chern_degree < 0:
        raise ValueError("max_chern_degree must be non-negative.")

    if component is not None and max_chern_degree < component:
        raise ValueError("max_chern_degree must be >= component.")

    return component, max_chern_degree


def _chern_character_from_chern_classes(bundle: VectorBundle, max_chern_degree: int) -> tuple[sp.Expr, ...]:
    """
    Compute a bundle's Chern character from its Chern classes.

    Let ``x_1, ..., x_r`` be the formal Chern roots of a bundle ``E``. Then

    ``c_k(E) = e_k(x_1, ..., x_r)``

    is the k-th elementary symmetric function, while

    ``p_k(E) = Σ_j x_j^k``

    is the k-th power sum. The homogeneous Chern-character components are

    ``ch_k(E) = p_k(E) / k!``.

    Newton's identities determine the power sums recursively:

    ``p_k =
    c_1 p_{k-1} - c_2 p_{k-2} + ... +
    (-1)^(k-2) c_{k-1} p_1 + (-1)^(k-1) k c_k``.

    Parameters
    ----------
    bundle : VectorBundle
        Bundle whose stored Chern classes are used.
    max_chern_degree : int
        Highest Chern-character degree to compute.

    Returns
    -------
    tuple of sympy.Expr
        Tuple

        ``(rank(E), ch_1(E), ..., ch_N(E))``

        expressed in terms of the stored Chern classes.
    """
    c = bundle.chern_classes

    ch: list[sp.Expr] = [sp.sympify(bundle.rank)]
    power_sums: list[sp.Expr | None] = [None] * (max_chern_degree + 1)

    for k in range(1, max_chern_degree + 1):
        pk = sum(
            (-1) ** (i - 1)
            * _component(c, i)
            * power_sums[k - i]
            for i in range(1, k)
        )

        pk += (-1) ** (k - 1) * k * _component(c, k)

        pk = sp.expand(pk)
        power_sums[k] = pk

        ch.append(sp.expand(pk / sp.factorial(k)))

    return tuple(ch)


def _chern_character_using_chern_classes(expr: sp.Expr, max_chern_degree: int) -> tuple[sp.Expr, ...]:
    """
    Compute a Chern character using base-bundle Chern classes as input.

    For every base vector bundle in ``expr``, the stored Chern classes are
    converted to Chern-character components using Newton's identities. The
    resulting components are supplied directly to the recursive Chern-character
    evaluator without modifying the bundle objects.

    Parameters
    ----------
    expr : sympy.Expr
        K-theory expression to evaluate.
    max_chern_degree : int
        Highest Chern-character degree to compute.

    Returns
    -------
    tuple of sympy.Expr
        Chern character of ``expr`` expressed in terms of the original bundles'
        stored Chern classes.

    Raises
    ------
    ValueError
        If the expression contains vector bundles over incompatible schemes.
    NotImplementedError
        If the expression contains an unsupported operation.
    """
    _validate_same_scheme(expr)
    return _compute_chern_character(expr, max_chern_degree, bundle_character_getter=_chern_character_from_chern_classes)


def _chern_classes_from_chern_character(ch: tuple[sp.Expr, ...], max_chern_degree: int) -> tuple[sp.Expr, ...]:
    """
    Recover Chern classes from homogeneous Chern-character components.

    Define the power sums by

    ``p_i = i! ch_i``.

    Newton's identities give ``c_0 = 1`` and, for ``k >= 1``,

    ``k c_k =
    Σ_{i=1}^k (-1)^(i-1) c_{k-i} p_i``.

    Equivalently,

    ``c_k =
    (1/k) Σ_{i=1}^k
    (-1)^(i-1) c_{k-i} i! ch_i``.

    Parameters
    ----------
    ch : tuple of sympy.Expr
        Homogeneous Chern-character components.
    max_chern_degree : int
        Highest Chern class to compute.

    Returns
    -------
    tuple of sympy.Expr
        Chern classes

        ``(1, c_1, ..., c_N)``.
    """
    c: list[sp.Expr] = [sp.Integer(1)]

    for k in range(1, max_chern_degree + 1):
        ck = sum(
            (-1) ** (i - 1)
            * c[k - i]
            * sp.factorial(i)
            * _component(ch, i)
            for i in range(1, k + 1)
        )

        c.append(sp.expand(ck / k))

    return tuple(c)


def chern_class(expr: sp.Expr, component: Optional[int] = None, *, max_chern_degree: Optional[int] = None, use_chern_classes: bool = True) -> tuple[sp.Expr, ...] | sp.Expr:
    """
    Compute the Chern classes of a symbolic K-theory expression.

    The computation proceeds through the Chern character:

    1. Compute ``ch(expr)`` using the K-theoretic rules for direct sums,
       tensor products, powers, symmetric powers, and exterior powers.
    2. Recover the Chern classes from ``ch(expr)`` using Newton's identities.

    When ``use_chern_classes=True``, the stored Chern classes of each base
    bundle are first converted to Chern-character components. Consequently,
    the final result is expressed in terms of symbols such as ``c1_E`` and
    ``c2_E``.

    Parameters
    ----------
    expr : sympy.Expr
        Expression built from vector bundles using sums, tensor products,
        powers, duals, homomorphism bundles, endomorphism bundles,
        determinants, symmetric powers, and exterior powers.
    component : int, optional
        Degree of the Chern class to return. If omitted, all components through
        ``max_chern_degree`` are returned.
    max_chern_degree : int, optional
        Highest Chern degree to compute. If omitted, it is inferred from the
        vector bundles in ``expr``.
    use_chern_classes : bool, default=True
        Select the primary characteristic data of the base bundles.

        - ``True`` uses their stored Chern classes and expresses the answer in
          terms of those classes.
        - ``False`` uses their stored Chern-character components directly.

    Returns
    -------
    tuple of sympy.Expr or sympy.Expr
        Full tuple

        ``(c_0(expr), c_1(expr), ..., c_N(expr))``

        or the selected Chern class.

    Raises
    ------
    TypeError
        If ``component`` is not an integer.
    ValueError
        If a degree is negative, if ``max_chern_degree < component``, or if
        bundles over incompatible schemes occur in the expression.
    NotImplementedError
        If the Chern-character backend encounters an unsupported expression.

    Notes
    -----
    The normalization is

    ``c_0(expr) = 1``.

    For a direct sum, the resulting components satisfy the Whitney product
    formula

    ``c(E ⊕ F) = c(E) c(F)``.

    Tensor products and power operations are handled by first using the
    multiplicative Chern character and then converting back to Chern classes.

    Examples
    --------
    Compute all inferred components:

    ``chern_class(E + F)``

    Compute only the second Chern class:

    ``chern_class(E * F, component=2)``

    Compute through degree three:

    ``chern_class(E.sym(2), max_chern_degree=3)``

    Use the stored Chern-character components as input:

    ``chern_class(E + F, use_chern_classes=False)``
    """
    expr = sp.sympify(expr)

    component, max_chern_degree = _validate_component_and_degree(
        expr=expr,
        component=component,
        max_chern_degree=max_chern_degree,
    )

    if use_chern_classes:
        ch = _chern_character_using_chern_classes(
            expr,
            max_chern_degree=max_chern_degree,
        )
    else:
        ch = chern_character(
            expr,
            max_chern_degree=max_chern_degree,
        )

    c = _chern_classes_from_chern_character(
        ch,
        max_chern_degree=max_chern_degree,
    )

    if component is not None:
        return c[component]

    return c