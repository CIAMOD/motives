"""Operations involving duals, Hom bundles, endomorphisms and determinants."""

import sympy as sp

from ..objects.determinant_bundle import DeterminantBundle
from ..objects.dual_bundle import DualBundle
from ..objects.vector_bundle import VectorBundle


def Dual(bundle: VectorBundle) -> VectorBundle:
    """Construct the dual of a vector bundle.

    For a vector bundle E, this operation returns a ``DualBundle`` representing
    E^∨. Double dualization is simplified automatically using

        Dual(Dual(E)) = E.

    Parameters
    ----------
    bundle : VectorBundle
        Vector bundle whose dual is constructed.

    Returns
    -------
    VectorBundle
        Original bundle when ``bundle`` is already a ``DualBundle``.
        Otherwise, a new ``DualBundle`` representing its dual.

    Raises
    ------
    TypeError
        If ``bundle`` is not a vector bundle.
    """
    if not isinstance(bundle, VectorBundle):
        raise TypeError("Dual expects a VectorBundle.")

    if isinstance(bundle, DualBundle):
        return bundle.bundle

    return DualBundle(bundle)


def Hom(E: VectorBundle, F: VectorBundle) -> sp.Expr:
    """Construct the homomorphism bundle from E to F.

    The homomorphism bundle is defined by

        Hom(E, F) = E^∨ tensor F.

    Multiplication represents tensor product in the symbolic K-theory
    expressions, so the result is represented as ``Dual(E) * F``.

    Parameters
    ----------
    E : VectorBundle
        Source vector bundle.
    F : VectorBundle
        Target vector bundle.

    Returns
    -------
    sympy.Expr
        Symbolic tensor product ``Dual(E) * F``.

    Raises
    ------
    TypeError
        If either argument is not a vector bundle.
    ValueError
        If the two bundles are defined over different schemes.
    """
    if not isinstance(E, VectorBundle) or not isinstance(F, VectorBundle):
        raise TypeError("Hom expects two VectorBundle objects.")

    if E.scheme is not F.scheme:
        raise ValueError("source and target must be defined over the same scheme.")

    return Dual(E) * F


def End(bundle: VectorBundle) -> sp.Expr:
    """Construct the endomorphism bundle of a vector bundle.

    The endomorphism bundle is

        End(E) = Hom(E, E) = E^∨ tensor E.

    Parameters
    ----------
    bundle : VectorBundle
        Vector bundle whose endomorphism bundle is constructed.

    Returns
    -------
    sympy.Expr
        Symbolic tensor product ``Dual(E) * E``.

    Raises
    ------
    TypeError
        If ``bundle`` is not a vector bundle.
    """
    if not isinstance(bundle, VectorBundle):
        raise TypeError("End expects a VectorBundle.")

    return Hom(bundle, bundle)


def Det(bundle: VectorBundle) -> DeterminantBundle:
    """Construct the determinant line bundle of a vector bundle.

    For a vector bundle E of rank r, the determinant is

        det(E) = Wedge^r(E).

    The result is represented by a ``DeterminantBundle`` of rank one, whose
    Chern classes and Chern character are determined by E.

    Parameters
    ----------
    bundle : VectorBundle
        Vector bundle whose determinant is constructed.

    Returns
    -------
    DeterminantBundle
        Determinant line bundle of ``bundle``.

    Raises
    ------
    TypeError
        If ``bundle`` is not a vector bundle.
    """
    if not isinstance(bundle, VectorBundle):
        raise TypeError("Det expects a VectorBundle.")

    return DeterminantBundle(bundle)