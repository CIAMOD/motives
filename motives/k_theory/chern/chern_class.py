from __future__ import annotations

from typing import Optional

import sympy as sp

from ..objects.vector_bundle import VectorBundle
from .chern_character import (
    chern_character,
    _component,
    _infer_max_chern_degree,
    _vector_bundles,
)


def _validate_component_and_degree(
    expr: sp.Expr,
    component: Optional[int],
    max_chern_degree: Optional[int],
) -> tuple[Optional[int], int]:
    """
    Validate the requested component and truncation degree.

    If component is provided and max_chern_degree is not, compute only up to
    that component. Otherwise, infer the degree from the bundles in the
    expression.
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


def _chern_character_from_chern_classes(
    bundle: VectorBundle,
    max_chern_degree: int,
) -> tuple[sp.Expr, ...]:
    """
    Compute the Chern character of a vector bundle from its Chern classes.

    If c_i are the elementary symmetric functions in the Chern roots and p_i
    are the power sums, then Newton's identities give:

        p_k = c_1 p_{k-1} - c_2 p_{k-2} + ... + (-1)^(k-1) k c_k

    Then:

        ch_k = p_k / k!
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


def _chern_character_using_chern_classes(
    expr: sp.Expr,
    max_chern_degree: int,
) -> tuple[sp.Expr, ...]:
    """
    Compute ch(expr), but using the Chern classes of the base vector bundles
    as the primary data.

    This temporarily replaces the stored Chern character of each VectorBundle
    by the one induced from its Chern classes, calls the existing
    chern_character backend, and then restores the original data.
    """
    bundles = _vector_bundles(expr)

    original_chern_characters = {
        bundle: bundle.chern_character
        for bundle in bundles
    }

    try:
        for bundle in bundles:
            bundle._chern_character = _chern_character_from_chern_classes(
                bundle,
                max_chern_degree,
            )

        return chern_character(
            expr,
            max_chern_degree=max_chern_degree,
        )

    finally:
        for bundle, original_chern_character in original_chern_characters.items():
            bundle._chern_character = original_chern_character


def _chern_classes_from_chern_character(
    ch: tuple[sp.Expr, ...],
    max_chern_degree: int,
) -> tuple[sp.Expr, ...]:
    """
    Compute Chern classes from the Chern character.

    If p_i = i! ch_i, Newton's identities give:

        c_0 = 1

        c_k = 1/k * sum(
            (-1)^(i-1) c_{k-i} p_i
            for i = 1, ..., k
        )
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


def chern_class(
    expr: sp.Expr,
    component: Optional[int] = None,
    *,
    max_chern_degree: Optional[int] = None,
    use_chern_classes: bool = True,
) -> tuple[sp.Expr, ...] | sp.Expr:
    """
    Compute the total Chern class of an expression of vector bundles.

    Parameters
    ----------
    expr
        Expression built from vector bundles using sums, products, powers,
        symmetric powers, and exterior powers.

    component
        If None, return the full tuple:

            (c_0, c_1, ..., c_max_chern_degree)

        If an integer is given, return only that component.

    max_chern_degree
        Maximum Chern degree to compute. If not provided, it is inferred from
        the vector bundles in the expression.

    use_chern_classes
        If True, the Chern classes of the base vector bundles are used as the
        primary input data. This gives answers in terms of c_i_E, c_i_F, ...

        If False, the stored Chern character components are used directly.
        This gives answers in terms of ch_i_E, ch_i_F, ...

    Examples
    --------
    chern_class(E + F)

    chern_class(E * F, component=2)

    chern_class(E.sym(2), max_chern_degree=3)

    chern_class(E + F, use_chern_classes=False)
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