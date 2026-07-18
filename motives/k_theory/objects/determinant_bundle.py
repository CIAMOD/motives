from __future__ import annotations

import sympy as sp

from .vector_bundle import VectorBundle


class DeterminantBundle(VectorBundle):
    """Represent the determinant line bundle of a vector bundle.

    For a vector bundle E of rank r, its determinant is its highest non-zero
    exterior power:

        det(E) = Wedge^r(E).

    The determinant is defined over the same scheme as E and always has rank
    one. Its Chern classes satisfy

        c0(det(E)) = 1,
        c1(det(E)) = c1(E),
        ci(det(E)) = 0 for every i >= 2.

    Since the determinant is a line bundle, its Chern character is determined
    by its degree-one component:

        ch(det(E)) = exp(ch1(E)),

    and therefore

        chi(det(E)) = ch1(E)^i / i!.

    The characteristic data are computed dynamically from E, so changes to the
    original bundle are automatically reflected in its determinant.

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
        """Create the symbolic object representing the determinant bundle."""
        if not isinstance(bundle, VectorBundle):
            raise TypeError("DeterminantBundle expects a VectorBundle.")

        return super().__new__(cls, f"Det({bundle})", bundle.scheme, rank=1, max_degree=bundle.max_degree)

    def __init__(self, bundle: VectorBundle):
        """Initialize the determinant from its original vector bundle."""
        if not isinstance(bundle, VectorBundle):
            raise TypeError("DeterminantBundle expects a VectorBundle.")

        self.bundle = bundle
        super().__init__(f"Det({bundle})", bundle.scheme, rank=1, max_degree=bundle.max_degree)

    @property
    def chern_classes(self) -> tuple[sp.Expr, ...]:
        """Return the Chern classes of the determinant line bundle.

        The only potentially non-zero positive-degree class is

            c1(det(E)) = c1(E).
        """
        if self.max_degree == 0:
            return (sp.Integer(1),)

        return (sp.Integer(1), self.bundle.c(1)) + (sp.Integer(0),) * (self.max_degree - 1)

    @chern_classes.setter
    def chern_classes(self, value) -> None:
        """Prevent independent assignment of determinant Chern classes.

        The value ``None`` is accepted internally while ``VectorBundle`` is
        initialized. Other values are rejected because the classes are
        determined by the original bundle.
        """
        if value is not None:
            raise AttributeError("The Chern classes of a DeterminantBundle are determined by its original bundle.")

    @property
    def chern_character(self) -> tuple[sp.Expr, ...]:
        """Return the Chern character of the determinant line bundle.

        Its components are computed using

            chi(det(E)) = ch1(E)^i / i!.
        """
        if self.max_degree == 0:
            return (sp.Integer(1),)

        ch1 = self.bundle.ch(1)
        return tuple(sp.Integer(1) if i == 0 else sp.expand(ch1**i / sp.factorial(i)) for i in range(self.max_degree + 1))

    @chern_character.setter
    def chern_character(self, value) -> None:
        """Prevent independent assignment of a determinant Chern character.

        The value ``None`` is accepted internally while ``VectorBundle`` is
        initialized. Other values are rejected because the character is
        determined by the original bundle.
        """
        if value is not None:
            raise AttributeError("The Chern character of a DeterminantBundle is determined by its original bundle.")

    def _sympystr(self, printer) -> str:
        """Return the plain-text representation ``Det(E)``."""
        return f"Det({printer.doprint(self.bundle)})"

    def _latex(self, printer) -> str:
        """Return the LaTeX representation of the determinant bundle."""
        return rf"\det\left({printer._print(self.bundle)}\right)"