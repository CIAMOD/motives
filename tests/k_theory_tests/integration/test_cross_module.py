import pytest
import sympy as sp

from motives.k_theory import Det, Dual, End, Hom, SymPower, VectorBundle, Wedge, chern_character, chern_class

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


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_determinant_matches_highest_exterior_power_for_consistent_bundle_data(scheme_factory, rank):
    """Verify Det(E) and the highest exterior power have identical invariants."""
    X = scheme_factory(dimension=3)
    roots = sp.symbols(f"x1:{rank + 1}")
    t = sp.Symbol("t")
    total_chern_class = sp.expand(sp.prod(1 + root * t for root in roots))
    classes = tuple(total_chern_class.coeff(t, degree) for degree in range(4))
    character = (sp.Integer(rank),) + tuple(
        sp.expand(sum(root**degree for root in roots) / sp.factorial(degree))
        for degree in range(1, 4)
    )
    E = VectorBundle(f"E_top_wedge_{rank}", X, rank=rank, chern_classes=classes, chern_character=character)

    assert_tuple_equal(chern_character(Det(E)), chern_character(Wedge(rank, E)))
    assert_tuple_equal(chern_class(Det(E)), chern_class(Wedge(rank, E)))
    assert_tuple_equal(
        chern_class(Det(E), use_chern_classes=False),
        chern_class(Wedge(rank, E), use_chern_classes=False)
    )


def test_dual_and_determinant_commute_at_invariant_level(explicit_bundles):
    """Verify det(E dual) and det(E) dual have the same Chern invariants."""
    _, E, *_ = explicit_bundles
    determinant_of_dual = Det(Dual(E))
    dual_of_determinant = Dual(Det(E))

    assert_tuple_equal(chern_character(determinant_of_dual), chern_character(dual_of_determinant))
    assert_tuple_equal(chern_class(determinant_of_dual), chern_class(dual_of_determinant))
    assert_tuple_equal(
        chern_class(determinant_of_dual, use_chern_classes=False),
        chern_class(dual_of_determinant, use_chern_classes=False)
    )


def test_nested_derived_bundles_work_inside_composite_expressions(explicit_bundles):
    """Verify nested derived bundles inside sums and tensor products."""
    _, E, F, ecs, fcs, eus, fvs = explicit_bundles
    e1, *_ = ecs
    f1, *_ = fcs
    u1, *_ = eus
    v1, *_ = fvs
    expression = Dual(E) + Det(F) * Dual(Det(E))

    assert chern_character(expression, component=0) == E.rank + 1
    assert_expr_equal(chern_character(expression, component=1), v1 - 2 * u1)
    assert_expr_equal(chern_class(expression, component=1), f1 - 2 * e1)
