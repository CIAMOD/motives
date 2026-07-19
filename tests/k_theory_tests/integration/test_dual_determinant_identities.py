"""Mathematical identity tests for DualBundle and DeterminantBundle."""

from itertools import combinations

import pytest
import sympy as sp

from motives.k_theory import Det, Dual, SymPower, VectorBundle, Wedge, chern_character, chern_class

from ..conftest import assert_tuple_equal


def _bundle_from_chern_roots(name, scheme, roots):
    """Create a bundle whose Chern data is induced by the supplied formal roots."""
    rank = len(roots)
    classes = [sp.Integer(1)]
    for degree in range(1, scheme.dimension + 1):
        classes.append(sum((sp.prod(items) for items in combinations(roots, degree)), sp.Integer(0)) if degree <= rank else sp.Integer(0))
    character = [sp.Integer(rank)]
    for degree in range(1, scheme.dimension + 1):
        character.append(sum((root**degree for root in roots), sp.Integer(0)) / sp.factorial(degree))
    return VectorBundle(name, scheme, rank=rank, chern_classes=tuple(classes), chern_character=tuple(character))


@pytest.mark.parametrize("rank", [2, 3])
def test_determinant_agrees_with_the_top_exterior_power(scheme_factory, rank):
    """Det(E) and the top exterior power of E must have identical Chern invariants."""
    X = scheme_factory(name=f"X_top_wedge_{rank}", dimension=3)
    roots = sp.symbols(" ".join(f"x_{rank}_{i}" for i in range(rank)))
    roots = (roots,) if rank == 1 else tuple(roots)
    E = _bundle_from_chern_roots(f"E_top_wedge_{rank}", X, roots)
    top_wedge = Wedge(E.rank, E)

    assert_tuple_equal(chern_character(Det(E)), chern_character(top_wedge))
    assert_tuple_equal(chern_class(Det(E)), chern_class(top_wedge))
    assert_tuple_equal(chern_class(Det(E), use_chern_classes=False), chern_class(top_wedge, use_chern_classes=False))


def test_determinant_commutes_with_dual_at_the_invariant_level(scheme_factory):
    """The identity det(E dual) = det(E) dual must hold for classes and characters."""
    X = scheme_factory(name="X_det_dual_identity", dimension=3)
    x, y, z = sp.symbols("x y z")
    E = _bundle_from_chern_roots("E_det_dual_identity", X, (x, y, z))
    left = Det(Dual(E))
    right = Dual(Det(E))

    assert_tuple_equal(chern_character(left), chern_character(right))
    assert_tuple_equal(chern_class(left), chern_class(right))
    assert_tuple_equal(chern_class(left, use_chern_classes=False), chern_class(right, use_chern_classes=False))


def test_determinant_of_a_line_bundle_has_the_same_invariants(scheme_factory):
    """The determinant of a rank-one bundle must agree with that line bundle."""
    X = scheme_factory(name="X_det_line", dimension=3)
    x = sp.symbols("x")
    L = _bundle_from_chern_roots("L_det_line", X, (x,))

    assert_tuple_equal(chern_character(Det(L)), chern_character(L))
    assert_tuple_equal(chern_class(Det(L)), chern_class(L))
    assert_tuple_equal(chern_class(Det(L), use_chern_classes=False), chern_class(L, use_chern_classes=False))


def test_determinant_is_idempotent_at_the_invariant_level(scheme_factory):
    """Applying determinant twice must preserve the determinant line bundle invariants."""
    X = scheme_factory(name="X_det_idempotent", dimension=3)
    x, y = sp.symbols("x y")
    E = _bundle_from_chern_roots("E_det_idempotent", X, (x, y))

    assert_tuple_equal(chern_character(Det(Det(E))), chern_character(Det(E)))
    assert_tuple_equal(chern_class(Det(Det(E))), chern_class(Det(E)))
    assert_tuple_equal(chern_class(Det(Det(E)), use_chern_classes=False), chern_class(Det(E), use_chern_classes=False))


def test_power_operations_accept_dual_and_determinant_children(scheme_factory):
    """Wedge and symmetric powers must recursively evaluate derived bundle children."""
    X = scheme_factory(name="X_derived_power_children", dimension=3)
    x, y = sp.symbols("x y")
    E = _bundle_from_chern_roots("E_derived_power_children", X, (x, y))
    expressions = (Wedge(2, Dual(E)), SymPower(2, Det(E)))

    for expression in expressions:
        assert_tuple_equal(expression.ch(), chern_character(expression))
        assert_tuple_equal(expression.c(), chern_class(expression))
        assert_tuple_equal(expression.c(use_chern_classes=False), chern_class(expression, use_chern_classes=False))
