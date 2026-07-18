import sympy as sp

from ..objects.determinant_bundle import DeterminantBundle
from ..objects.vector_bundle import VectorBundle
from ..objects.dual_bundle import Dual


def Hom(E: sp.Expr, F: sp.Expr) -> sp.Expr:
    """
    Construct the homomorphism bundle from ``E`` to ``F``.

    Mathematically,

    ``Hom(E, F) = E^∨ ⊗ F``.

    In the symbolic K-theory representation, multiplication denotes tensor
    product, so this function returns ``Dual(E) * F``.

    Parameters
    ----------
    E
        Source vector bundle or vector-bundle expression.
    F
        Target vector bundle or vector-bundle expression.

    Returns
    -------
    sympy.Expr
        Symbolic expression representing ``E^∨ ⊗ F``.
    """
    E = sp.sympify(E)
    F = sp.sympify(F)

    if E.scheme is not F.scheme:
        raise ValueError("source and target must be defined over the same scheme.")

    return Dual(E) * F


def End(E: sp.Expr) -> sp.Expr:
    """
    Construct the endomorphism bundle of ``E``.

    Mathematically,

    ``End(E) = Hom(E, E) = E^∨ ⊗ E``.

    Parameters
    ----------
    E
        Vector bundle or vector-bundle expression.

    Returns
    -------
    sympy.Expr
        Symbolic expression representing ``E^∨ ⊗ E``.
    """
    E = sp.sympify(E)

    return Hom(E, E)


def Det(bundle: VectorBundle) -> DeterminantBundle:
    """
    Construct the determinant line bundle of a vector bundle.

    For a vector bundle E of rank r, this operation returns the line bundle

        det(E) = Wedge^r(E).

    The result is represented by a ``DeterminantBundle`` rather than by a generic
    formal exterior-power expression. This allows its rank, Chern classes and
    Chern character components to be accessed directly.

    Parameters
    ----------
    bundle : VectorBundle
        Vector bundle whose determinant line bundle is constructed.

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