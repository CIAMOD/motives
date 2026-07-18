import pytest
import sympy as sp

from motives.k_theory import VectorBundle, exact_sequence_realtions, solve_exact_sequence
from motives.k_theory.operations.exact_sequences import _get_solve_variables, _validate_exact_sequence

from ..conftest import assert_expr_equal


def test_validate_exact_sequence_normalizes_terms(three_bundles):
    """Verify that validate exact sequence normalizes terms."""
    _, E, F, G = three_bundles
    assert _validate_exact_sequence([E, F, G]) == (E, F, G)


@pytest.mark.parametrize("sequence", ["EF", 3, None])
def test_validate_exact_sequence_requires_ordered_sequence(sequence):
    """Verify that validate exact sequence requires ordered sequence."""
    with pytest.raises(TypeError, match="ordered sequence"):
        _validate_exact_sequence(sequence)


def test_validate_exact_sequence_requires_at_least_two_terms(bundle_factory):
    """Verify that validate exact sequence requires at least two terms."""
    with pytest.raises(ValueError, match="at least two"):
        _validate_exact_sequence([bundle_factory()])


def test_validate_exact_sequence_requires_bundle_atom():
    """Verify that validate exact sequence requires bundle atom."""
    with pytest.raises(ValueError, match="at least one VectorBundle"):
        _validate_exact_sequence([sp.Symbol("x"), sp.Symbol("y")])


def test_validate_exact_sequence_rejects_mixed_schemes(bundle_factory):
    """Verify that validate exact sequence rejects mixed schemes."""
    with pytest.raises(ValueError, match="same scheme"):
        _validate_exact_sequence([bundle_factory(), bundle_factory()])


def test_chern_character_relations_for_short_exact_sequence(three_bundles):
    """Verify that Chern character relations for short exact sequence."""
    _, E, F, G = three_bundles
    equations = exact_sequence_realtions([E, F, G])
    assert len(equations) == 4
    for degree, equation in enumerate(equations):
        assert equation == sp.Eq(sp.expand(E.ch(degree) + G.ch(degree)), sp.expand(F.ch(degree)), evaluate=False)


def test_component_relation_returns_one_equation(three_bundles):
    """Verify that component relation returns one equation."""
    _, E, F, G = three_bundles
    equation = exact_sequence_realtions([E, F, G], component=2)
    assert isinstance(equation, sp.Equality)
    assert equation == sp.Eq(E.ch(2) + G.ch(2), F.ch(2), evaluate=False)


def test_chern_class_relations_are_whitney_relations(three_bundles):
    """Verify that Chern class relations are whitney relations."""
    _, E, F, G = three_bundles
    equations = exact_sequence_realtions([E, F, G], invariant="chern_classes")
    assert len(equations) == 3
    assert equations[0] == sp.Eq(E.c(1) + G.c(1), F.c(1), evaluate=False)
    assert_expr_equal(equations[1].lhs, E.c(2) + E.c(1) * G.c(1) + G.c(2))
    assert equations[1].rhs == F.c(2)


def test_four_term_sequence_uses_alternating_even_odd_relation(bundle_factory):
    """Verify that four term sequence uses alternating even odd relation."""
    E = bundle_factory()
    X = E.scheme
    F = bundle_factory(scheme=X)
    G = bundle_factory(scheme=X)
    H = bundle_factory(scheme=X)
    equation = exact_sequence_realtions([E, F, G, H], component=1)
    assert equation == sp.Eq(E.ch(1) + G.ch(1), F.ch(1) + H.ch(1), evaluate=False)


def test_invalid_invariant_is_rejected(three_bundles):
    """Verify that invalid invariant is rejected."""
    _, E, F, G = three_bundles
    with pytest.raises(ValueError, match="invariant must be"):
        exact_sequence_realtions([E, F, G], invariant="invalid")


def test_get_solve_variables_expands_bundle_components(three_bundles):
    """Verify that get solve variables expands bundle components."""
    _, _, _, G = three_bundles
    assert _get_solve_variables(G, "chern_character", None) == tuple(value for value in G.ch() if isinstance(value, sp.Symbol))
    assert _get_solve_variables(G, "chern_classes", None) == tuple(value for value in G.c()[1:] if isinstance(value, sp.Symbol))
    assert _get_solve_variables(G, "chern_character", 2) == (G.ch(2),)


def test_get_solve_variables_accepts_symbols_deduplicates_and_rejects_invalid_targets(three_bundles):
    """Verify that get solve variables accepts symbols deduplicates and rejects invalid targets."""
    _, _, _, G = three_bundles
    x = sp.Symbol("x")
    assert _get_solve_variables([x, x], "chern_character", None) == (x,)
    with pytest.raises(TypeError, match="solve_for"):
        _get_solve_variables("x", "chern_character", None)
    with pytest.raises(TypeError, match="Every solve target"):
        _get_solve_variables([G, 2], "chern_character", None)


def test_get_solve_variables_rejects_bundle_with_only_numeric_requested_data(scheme_factory):
    """Verify that get solve variables rejects bundle with only numeric requested data."""
    X = scheme_factory(dimension=1)
    E = VectorBundle("E_numeric", X, rank=1, chern_classes=(1, 0), chern_character=(1, 0))
    with pytest.raises(ValueError, match="No symbolic variables"):
        _get_solve_variables(E, "chern_character", None)


def test_solve_short_exact_sequence_for_unknown_character(three_bundles):
    """Verify that solve short exact sequence for unknown character."""
    _, E, F, G = three_bundles
    solution = solve_exact_sequence([E, F, G], G)
    assert len(solution) == 1
    assert_expr_equal(solution[0][G.ch(1)], F.ch(1) - E.ch(1))
    assert_expr_equal(solution[0][G.ch(2)], F.ch(2) - E.ch(2))
    assert_expr_equal(solution[0][G.ch(3)], F.ch(3) - E.ch(3))


def test_solve_one_component(three_bundles):
    """Verify that solve one component."""
    _, E, F, G = three_bundles
    solution = solve_exact_sequence([E, F, G], G, component=2)
    assert solution == [{G.ch(2): -E.ch(2) + F.ch(2)}]


def test_solve_chern_classes_for_unknown_bundle(three_bundles):
    """Verify that solve Chern classes for unknown bundle."""
    _, E, F, G = three_bundles
    solution = solve_exact_sequence([E, F, G], G, invariant="chern_classes")
    assert len(solution) == 1
    assert_expr_equal(solution[0][G.c(1)], F.c(1) - E.c(1))
    assert_expr_equal(solution[0][G.c(2)], F.c(2) - E.c(2) - E.c(1) * solution[0][G.c(1)])


def test_multiple_sequences_are_solved_together(three_bundles):
    """Verify that multiple sequences are solved together."""
    _, E, F, G = three_bundles
    solution = solve_exact_sequence([[E, F, G], [E, F, G]], G, component=1)
    assert len(solution) == 1
    assert_expr_equal(solution[0][G.ch(1)], F.ch(1) - E.ch(1))


def test_solve_accepts_explicit_symbol(three_bundles):
    """Verify that solve accepts explicit symbol."""
    _, E, F, G = three_bundles
    solution = solve_exact_sequence([E, F, G], G.ch(1), component=1)
    assert solution == [{G.ch(1): -E.ch(1) + F.ch(1)}]


def test_solve_validation_errors(three_bundles):
    """Verify that solve validation errors."""
    _, E, F, G = three_bundles
    with pytest.raises(ValueError, match="At least one"):
        solve_exact_sequence([], G)
    with pytest.raises(ValueError, match="do not appear"):
        solve_exact_sequence([E, F, G], sp.Symbol("unrelated"))
    with pytest.raises(ValueError, match="non-trivial equations"):
        solve_exact_sequence([E, E], E.ch(1), component=1)
