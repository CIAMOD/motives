from __future__ import annotations

import sympy as sp

from .vector_bundle import VectorBundle


class DualBundle(VectorBundle):
    """Represent the dual of a vector bundle.

    For a vector bundle E, its dual E^∨ is defined over the same scheme and
    has the same rank:

        rank(E^∨) = rank(E).

    Dualization changes the sign of every Chern root. Therefore, its Chern
    classes and homogeneous Chern-character components satisfy

        ci(E^∨) = (-1)^i ci(E),
        chi(E^∨) = (-1)^i chi(E).

    The characteristic data of the dual are computed dynamically from the
    original bundle. Consequently, changes to the Chern classes or Chern
    character of E are immediately reflected in E^∨.

    Parameters
    ----------
    bundle : VectorBundle
        Vector bundle whose dual is represented.

    Raises
    ------
    TypeError
        If ``bundle`` is not a vector bundle.
    """

    def __new__(cls, bundle: VectorBundle):
        """Create the symbolic object representing the dual bundle."""
        if not isinstance(bundle, VectorBundle):
            raise TypeError("DualBundle expects a VectorBundle.")

        return super().__new__(cls, f"Dual({bundle})", bundle.scheme, rank=bundle.rank, max_degree=bundle.max_degree)

    def __init__(self, bundle: VectorBundle):
        """Initialize the dual bundle from its original vector bundle."""
        if not isinstance(bundle, VectorBundle):
            raise TypeError("DualBundle expects a VectorBundle.")

        self.bundle = bundle
        super().__init__(f"Dual({bundle})", bundle.scheme, rank=bundle.rank, max_degree=bundle.max_degree)

    @property
    def chern_classes(self) -> tuple[sp.Expr, ...]:
        """Return the Chern classes of the dual bundle.

        The components are computed using

            ci(E^∨) = (-1)^i ci(E).
        """
        return tuple(sp.Integer(-1) ** i * self.bundle.c(i) for i in range(self.max_degree + 1))

    @chern_classes.setter
    def chern_classes(self, value) -> None:
        """Prevent independent assignment of Chern classes to the dual bundle.

        The value ``None`` is accepted internally while ``VectorBundle`` is
        initialized. All other values are rejected because the classes of the
        dual are determined by the original bundle.
        """
        if value is not None:
            raise AttributeError("The Chern classes of a DualBundle are determined by its original bundle.")

    @property
    def chern_character(self) -> tuple[sp.Expr, ...]:
        """Return the Chern character of the dual bundle.

        The components are computed using

            chi(E^∨) = (-1)^i chi(E).
        """
        return tuple(sp.Integer(-1) ** i * self.bundle.ch(i) for i in range(self.max_degree + 1))

    @chern_character.setter
    def chern_character(self, value) -> None:
        """Prevent independent assignment of a Chern character to the dual.

        The value ``None`` is accepted internally while ``VectorBundle`` is
        initialized. All other values are rejected because the character of
        the dual is determined by the original bundle.
        """
        if value is not None:
            raise AttributeError("The Chern character of a DualBundle is determined by its original bundle.")

    def _sympystr(self, printer) -> str:
        """Return the plain-text representation ``Dual(E)``."""
        return f"Dual({printer.doprint(self.bundle)})"

    def _latex(self, printer) -> str:
        """Return the LaTeX representation of the dual bundle."""
        return rf"\left({printer._print(self.bundle)}\right)^\vee"