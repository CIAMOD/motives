import sympy as sp

from motives.k_theory import Det, Dual, DualBundle, exact_sequence_relations, solve_exact_sequence


def test_regression_dual_is_vector_bundle_with_direct_chern_access(bundle_factory):
    """Verify that regression dual is vector bundle with direct Chern access."""
    E = bundle_factory()
    D = Dual(E)
    assert isinstance(D, DualBundle)
    assert D.c(1) == -E.c(1)
    assert D.ch(2) == E.ch(2)


def test_regression_double_dual_is_original_object(bundle_factory):
    """Verify that regression double dual is original object."""
    E = bundle_factory()
    assert Dual(Dual(E)) is E


def test_regression_determinant_is_not_generic_wedge_node(bundle_factory):
    """Verify that regression determinant is not generic wedge node."""
    from motives.k_theory import DeterminantBundle, Wedge
    determinant = Det(bundle_factory())
    assert isinstance(determinant, DeterminantBundle)
    assert not isinstance(determinant, Wedge)


def test_regression_exact_sequence_solver_returns_dictionary_solution(three_bundles):
    """Verify that regression exact sequence solver returns dictionary solution."""
    _, E, F, G = three_bundles
    solutions = solve_exact_sequence([E, F, G], G, component=1)
    assert solutions == [{G.ch(1): -E.ch(1) + F.ch(1)}]


def test_regression_exact_sequence_function_uses_current_public_name(three_bundles):
    """Verify that regression exact sequence function uses current public name."""
    _, E, F, G = three_bundles
    relation = exact_sequence_relations([E, F, G], component=1)
    assert isinstance(relation, sp.Equality)


def test_regression_expression_methods_are_installed_once_imported(bundle_factory):
    """Verify that regression expression methods are installed once imported."""
    E = bundle_factory()
    F = bundle_factory(scheme=E.scheme)
    expression = E + F
    assert expression.ch(1) == E.ch(1) + F.ch(1)
    assert expression.c(1) == E.c(1) + F.c(1)


def test_regression_chern_character_setter_exists_and_updates(bundle_factory):
    """Verify that regression Chern character setter exists and updates."""
    E = bundle_factory(dimension=2)
    x = sp.Symbol("x")
    E.chern_character = {1: x}
    assert E.ch(1) == x


def test_regression_scheme_name_validation_occurs_for_invalid_values():
    """Verify that regression scheme name validation occurs for invalid values."""
    from motives.k_theory import Scheme
    for invalid in ("", None):
        try:
            Scheme(invalid, 1)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid scheme name was accepted")
