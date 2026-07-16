from __future__ import annotations

import sympy as sp

from .vector_bundle import VectorBundle


class DeterminantBundle(VectorBundle):
    """
    Represent the determinant line bundle of a vector bundle.

    For a vector bundle E of rank r, its determinant is defined as its highest
    non-zero exterior power:

        det(E) = Wedge^r(E).

    The determinant is a vector bundle over the same scheme as E and always has
    rank one.

    Its Chern classes satisfy

        c0(det(E)) = 1,
        c1(det(E)) = c1(E),
        ci(det(E)) = 0 for every i >= 2.

    Because the determinant is a line bundle, its Chern character is determined
    entirely by its first Chern class:

        ch(det(E)) = exp(c1(E)).

    Therefore, its degree-i component is

        chi(det(E)) = c1(E)^i / i!.

    Parameters
    ----------
    bundle : VectorBundle
        Vector bundle whose determinant line bundle is represented.

    Raises
    ------
    TypeError
        If ``bundle`` is not a vector bundle.
    """

    def __new__(cls, bundle: VectorBundle):
        if not isinstance(bundle, VectorBundle):
            raise TypeError("DeterminantBundle expects a VectorBundle.")

        name = f"Det({bundle})"
        return super().__new__(cls, name, bundle.scheme, rank=1, max_degree=bundle.max_chern_degree)

    def __init__(self, bundle: VectorBundle):
        if not isinstance(bundle, VectorBundle):
            raise TypeError("DeterminantBundle expects a VectorBundle.")

        self.bundle = bundle
        max_degree = bundle.max_chern_degree

        chern_classes = [sp.Integer(1)] + [sp.Integer(0)] * max_degree
        chern_character = [sp.Integer(1)] + [sp.Integer(0)] * max_degree

        if max_degree >= 1:
            chern_classes[1] = bundle.c(1)
            ch1 = bundle.ch(1)

            for i in range(1, max_degree + 1):
                chern_character[i] = sp.expand(ch1**i / sp.factorial(i))

        super().__init__(
            f"Det({bundle})",
            bundle.scheme,
            rank=1,
            max_degree=max_degree,
            chern_classes=chern_classes,
            chern_character=chern_character,
        )