"""Integration tests for Dual and determinant bundles in the Chern engines."""

import pytest
import sympy as sp

from motives.k_theory import Det, Dual, chern_character, chern_class

from ..conftest import assert_expr_equal, assert_tuple_equal


def test_dual_chern_class_uses_the_selected_source_invariant(bundle_factory):
    """The dual formula must work both from stored classes and stored characters."""
    e1, e2, e3, u1, u2, u3 = sp.symbols("e1 e2 e3 u1 u2 u3")
    E = bundle_factory("E_dual_source", dimension=3, rank=4, chern_classes=(1, e1, e2, e3), chern_character=(4, u1, u2, u3))

    assert_tuple_equal(chern_class(Dual(E)), (1, -e1, e2, -e3))
    assert_tuple_equal(chern_class(Dual(E), use_chern_classes=False), (1, -u1, u1**2 / 2 - u2, -u1**3 / 6 + u1 * u2 - 2 * u3))


def test_determinant_chern_class_uses_the_selected_source_invariant(bundle_factory):
    """The determinant must retain only the first Chern class on either data path."""
    e1, e2, e3, u1, u2, u3 = sp.symbols("e1 e2 e3 u1 u2 u3")
    E = bundle_factory("E_det_source", dimension=3, rank=4, chern_classes=(1, e1, e2, e3), chern_character=(4, u1, u2, u3))

    assert_tuple_equal(chern_class(Det(E)), (1, e1, 0, 0))
    assert_tuple_equal(chern_class(Det(E), use_chern_classes=False), (1, u1, 0, 0))


def test_dual_and_determinant_support_components_and_explicit_truncation(bundle_factory):
    """Derived bundles must support component lookup and degrees above their stored input."""
    u1, u2, u3 = sp.symbols("u1 u2 u3")
    E = bundle_factory("E_truncation", dimension=3, rank=3, chern_character=(3, u1, u2, u3))

    assert_expr_equal(chern_character(Dual(E), component=0), 3)
    assert_expr_equal(chern_character(Dual(E), component=1), -u1)
    assert_expr_equal(chern_character(Dual(E), component=2), u2)
    assert_expr_equal(chern_character(Dual(E), component=3), -u3)
    assert_tuple_equal(chern_character(Dual(E), max_chern_degree=5), (3, -u1, u2, -u3, 0, 0))

    assert_expr_equal(chern_character(Det(E), component=3), u1**3 / 6)
    assert_tuple_equal(chern_character(Det(E), max_chern_degree=5), (1, u1, u1**2 / 2, u1**3 / 6, u1**4 / 24, u1**5 / 120))
    assert_tuple_equal(chern_class(Det(E), max_chern_degree=5, use_chern_classes=False), (1, u1, 0, 0, 0, 0))


def test_nested_dual_and_determinant_work_inside_arbitrary_supported_expressions(bundle_factory):
    """Nested derived bundles must remain valid inside sums and tensor products."""
    E = bundle_factory("E_nested_derived", dimension=3, rank=3)
    F = bundle_factory("F_nested_derived", scheme=E.scheme, rank=2)
    expression = Dual(E) + Det(F) * Dual(Det(E))

    assert_tuple_equal(expression.ch(), chern_character(expression))
    assert_tuple_equal(expression.c(), chern_class(expression))
    assert_tuple_equal(expression.c(use_chern_classes=False), chern_class(expression, use_chern_classes=False))


def test_nested_derived_bundles_preserve_same_scheme_validation(bundle_factory):
    """Wrapping bundles in Dual or Det must not bypass the same-scheme requirement."""
    E = bundle_factory("E_nested_scheme", dimension=2, rank=2)
    F = bundle_factory("F_nested_scheme", dimension=2, rank=2)
    expression = Dual(E) + Det(F)

    with pytest.raises(ValueError, match="same scheme"):
        chern_character(expression)

    with pytest.raises(ValueError, match="same scheme"):
        chern_class(expression)


def test_chern_class_restores_source_character_after_success_and_failure(bundle_factory):
    """Temporary class-to-character conversion must never mutate the source bundle permanently."""
    e1, e2, e3, u1, u2, u3 = sp.symbols("e1 e2 e3 u1 u2 u3")
    E = bundle_factory("E_restore_character", dimension=3, rank=3, chern_classes=(1, e1, e2, e3), chern_character=(3, u1, u2, u3))
    original_character = E.ch()

    chern_class(Dual(Det(E)))
    assert_tuple_equal(E.ch(), original_character)

    with pytest.raises(NotImplementedError):
        chern_class(sp.sin(Dual(E)))

    assert_tuple_equal(E.ch(), original_character)
