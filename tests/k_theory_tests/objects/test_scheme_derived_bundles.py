"""Tests for the derived standard bundles attached to a Scheme."""

import sympy as sp

from motives.k_theory import Det, DeterminantBundle, Dual, DualBundle, chern_character, chern_class

from ..conftest import assert_expr_equal, assert_tuple_equal


def test_scheme_internal_derived_bundles_agree_with_the_public_factories(scheme_factory):
    """Scheme may construct internal classes directly while exposing factory-equivalent objects."""
    X = scheme_factory(name="X_scheme_factory_agreement", dimension=3)

    assert isinstance(X.Tx_dual, DualBundle)
    assert isinstance(X.Kx, DeterminantBundle)
    assert Dual(X.Tx) == X.Tx_dual
    assert Det(X.Tx_dual) == X.Kx
    assert Dual(X.Tx_dual) is X.Tx


def test_tangent_updates_propagate_through_cotangent_and_canonical_bundles(scheme_factory):
    """Both stored Chern representations must propagate through Tx_dual and Kx."""
    X = scheme_factory(name="X_scheme_dynamic_derived", dimension=3)
    c1, c2, c3, ch1, ch2, ch3 = sp.symbols("c1 c2 c3 ch1 ch2 ch3")
    X.Tx.chern_classes = {1: c1, 2: c2, 3: c3}
    X.Tx.chern_character = {1: ch1, 2: ch2, 3: ch3}

    assert_tuple_equal(X.Tx_dual.c(), (1, -c1, c2, -c3))
    assert_tuple_equal(X.Tx_dual.ch(), (3, -ch1, ch2, -ch3))
    assert_tuple_equal(X.Kx.c(), (1, -c1, 0, 0))
    assert_tuple_equal(X.Kx.ch(), (1, -ch1, ch1**2 / 2, -ch1**3 / 6))


def test_chern_engines_agree_with_scheme_derived_bundle_properties(scheme_factory):
    """The standalone Chern engines and object methods must agree for Tx_dual and Kx."""
    X = scheme_factory(name="X_scheme_chern_engines", dimension=3)
    c1, c2, c3, ch1, ch2, ch3 = sp.symbols("c1 c2 c3 ch1 ch2 ch3")
    X.Tx.chern_classes = {1: c1, 2: c2, 3: c3}
    X.Tx.chern_character = {1: ch1, 2: ch2, 3: ch3}

    for bundle in (X.Tx_dual, X.Kx):
        assert_tuple_equal(chern_character(bundle), bundle.ch())
        assert_tuple_equal(chern_class(bundle), bundle.c())

    assert_tuple_equal(chern_class(X.Tx_dual, use_chern_classes=False), (1, -ch1, ch1**2 / 2 - ch2, -ch1**3 / 6 + ch1 * ch2 - 2 * ch3))
    assert_tuple_equal(chern_class(X.Kx, use_chern_classes=False), (1, -ch1, 0, 0))


def test_canonical_bundle_has_the_expected_first_class_and_character(scheme_factory):
    """Kx = det(Tx dual) must satisfy c1(Kx) = -c1(Tx) and ch(Kx) = exp(-ch1(Tx))."""
    X = scheme_factory(name="X_scheme_canonical_identity", dimension=3)
    c1, ch1 = sp.symbols("c1 ch1")
    X.Tx.chern_classes = {1: c1}
    X.Tx.chern_character = {1: ch1}

    assert_expr_equal(X.Kx.c(1), -X.Tx.c(1))
    assert_expr_equal(X.Kx.c(2), 0)
    assert_expr_equal(X.Kx.c(3), 0)
    assert_tuple_equal(X.Kx.ch(), (1, -ch1, ch1**2 / 2, -ch1**3 / 6))
