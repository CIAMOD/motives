import pytest
import sympy as sp

from motives.k_theory import Det, Dual, End, Hom, SymPower, Wedge, chern_character, chern_class

from ..conftest import assert_expr_equal, assert_tuple_equal



def test_dual_character_and_class_function_agree_with_bundle_properties(explicit_bundles):
    """Verify that dual character and class function agree with bundle properties."""
    _, E, *_ = explicit_bundles
    D = Dual(E)
    assert_tuple_equal(chern_character(D), D.ch())
    assert_tuple_equal(chern_class(D), D.c())


def test_determinant_character_and_class_function_agree_with_properties(explicit_bundles):
    """Verify that determinant character and class function agree with properties."""
    _, E, *_ = explicit_bundles
    determinant = Det(E)
    assert_tuple_equal(chern_character(determinant), determinant.ch())
    assert_tuple_equal(chern_class(determinant), determinant.c())


def test_hom_character_equals_product_of_dual_and_target(explicit_bundles):
    """Verify that Hom character equals product of dual and target."""
    _, E, F, *_ = explicit_bundles
    assert_tuple_equal(chern_character(Hom(E, F)), chern_character(Dual(E) * F))


def test_endomorphism_bundle_has_rank_square(explicit_bundles):
    """Verify that endomorphism bundle has rank square."""
    _, E, *_ = explicit_bundles
    assert chern_character(End(E), component=0) == E.rank**2
    assert_expr_equal(chern_character(End(E), component=1), 0)


def test_standard_bundle_relations_on_scheme(scheme_factory):
    """Verify that standard bundle relations on scheme."""
    X = scheme_factory(dimension=3)
    assert Dual(X.Tx) == X.Tx_dual
    assert Det(X.Tx_dual) == X.Kx
    assert X.Kx.c(1) == -X.Tx.c(1)


def test_power_object_and_method_paths_have_same_character(explicit_bundles):
    """Verify that power object and method paths have same character."""
    _, E, *_ = explicit_bundles
    assert_tuple_equal(chern_character(E.sym(2)), chern_character(SymPower(2, E)))
    assert_tuple_equal(chern_character(E.wedge(2)), chern_character(Wedge(2, E)))


def test_expanded_and_unexpanded_direct_sum_powers_have_same_character(explicit_bundles):
    """Verify that expanded and unexpanded direct sum powers have same character."""
    _, E, F, *_ = explicit_bundles
    assert_tuple_equal(chern_character((E + F).wedge(2)), chern_character((E + F).to_wedge(2)))
    assert_tuple_equal(chern_character((E + F).sym(2)), chern_character((E + F).to_sym(2)))


def test_mutating_source_bundle_propagates_through_dual_determinant_and_composites(bundle_factory):
    """Verify that mutating source bundle propagates through dual determinant and composites."""
    E = bundle_factory(dimension=2)
    D = Dual(E)
    determinant = Det(E)
    x, y = sp.symbols("x y")
    E.chern_classes = {1: x}
    E.chern_character = {1: y}
    assert D.c(1) == -x
    assert D.ch(1) == -y
    assert determinant.c(1) == x
    assert determinant.ch(1) == y
    assert chern_character(End(E), component=1) == 0
