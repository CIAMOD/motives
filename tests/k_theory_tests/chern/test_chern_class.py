import pytest
import sympy as sp

from motives.k_theory import Det, Dual, chern_class

from ..conftest import assert_expr_equal, assert_tuple_equal


def test_single_bundle_returns_stored_classes_by_default(explicit_bundles):
    """Verify that single bundle returns stored classes by default."""
    _, E, *_ = explicit_bundles

    assert_tuple_equal(chern_class(E), E.c())


def test_component_selection(explicit_bundles):
    """Verify that component selection."""
    _, E, *_ = explicit_bundles

    assert_expr_equal(chern_class(E, component=2), E.c(2))


def test_using_character_data_instead_of_class_data(explicit_bundles):
    """Verify that using character data instead of class data."""
    _, E, _, _, _, eus, _ = explicit_bundles
    u1, u2, u3 = eus
    expected = (1, u1, u1**2 / 2 - u2, u1**3 / 6 - u1 * u2 + 2 * u3)

    assert_tuple_equal(chern_class(E, use_chern_classes=False), expected)


def test_whitney_sum_formula_for_direct_sum(explicit_bundles):
    """Verify that whitney sum formula for direct sum."""
    _, E, F, ecs, fcs, *_ = explicit_bundles
    e1, e2, e3 = ecs
    f1, f2, f3 = fcs
    expected = (
        1,
        e1 + f1,
        e2 + e1 * f1 + f2,
        e3 + e2 * f1 + e1 * f2 + f3
    )

    assert_tuple_equal(chern_class(E + F), expected)


def test_virtual_difference_uses_inverse_total_chern_class(explicit_bundles):
    """Verify that virtual difference uses inverse total Chern class."""
    _, E, F, ecs, fcs, *_ = explicit_bundles
    e1, e2, e3 = ecs
    f1, f2, f3 = fcs
    expected_c1 = e1 - f1
    expected_c2 = e2 - e1 * f1 + f1**2 - f2
    expected_c3 = e3 - e2 * f1 + e1 * (f1**2 - f2) - f1**3 + 2 * f1 * f2 - f3

    assert_tuple_equal(chern_class(E - F), (1, expected_c1, expected_c2, expected_c3))


def test_tensor_product_first_class_formula(explicit_bundles):
    """Verify that tensor product first class formula."""
    _, E, F, ecs, fcs, *_ = explicit_bundles
    e1, *_ = ecs
    f1, *_ = fcs

    assert_expr_equal(chern_class(E * F, component=1), F.rank * e1 + E.rank * f1)


def test_dual_classes_alternate_sign(explicit_bundles):
    """Verify that dual classes alternate sign."""
    _, E, _, ecs, *_ = explicit_bundles
    e1, e2, e3 = ecs

    assert_tuple_equal(chern_class(Dual(E)), (1, -e1, e2, -e3))


def test_determinant_has_only_first_chern_class(explicit_bundles):
    """Verify that determinant has only first Chern class."""
    _, E, _, ecs, *_ = explicit_bundles
    e1, _, _ = ecs

    assert_tuple_equal(chern_class(Det(E)), (1, e1, 0, 0))


def test_dual_classes_can_be_recovered_from_character_data(explicit_bundles):
    """Verify dual classes when the stored Chern character is the primary input."""
    _, E, _, _, _, eus, _ = explicit_bundles
    u1, u2, u3 = eus
    expected = (1, -u1, u1**2 / 2 - u2, -u1**3 / 6 + u1 * u2 - 2 * u3)

    assert_tuple_equal(chern_class(Dual(E), use_chern_classes=False), expected)


def test_determinant_classes_can_be_recovered_from_character_data(explicit_bundles):
    """Verify determinant classes when the stored Chern character is used."""
    _, E, _, _, _, eus, _ = explicit_bundles
    u1, _, _ = eus

    assert_tuple_equal(chern_class(Det(E), use_chern_classes=False), (1, u1, 0, 0))


def test_nested_derived_bundle_classes_use_both_input_paths(explicit_bundles):
    """Verify nested dual and determinant classes from classes and characters."""
    _, E, _, ecs, _, eus, _ = explicit_bundles
    e1, _, _ = ecs
    u1, _, _ = eus

    assert_tuple_equal(chern_class(Det(Dual(E))), (1, -e1, 0, 0))
    assert_tuple_equal(chern_class(Dual(Det(E))), (1, -e1, 0, 0))
    assert_tuple_equal(chern_class(Det(Dual(E)), use_chern_classes=False), (1, -u1, 0, 0))
    assert_tuple_equal(chern_class(Dual(Det(E)), use_chern_classes=False), (1, -u1, 0, 0))


def test_derived_bundle_class_component_selection_and_truncation(explicit_bundles):
    """Verify component selection and truncation for derived bundle classes."""
    _, E, _, ecs, *_ = explicit_bundles
    e1, e2, _ = ecs

    assert chern_class(Det(E), component=2) == 0
    assert_tuple_equal(chern_class(Dual(E), max_chern_degree=2), (1, -e1, e2))


def test_scalar_total_chern_class_is_one():
    """Verify that scalar total Chern class is one."""
    a = sp.Symbol("a")

    assert chern_class(a, max_chern_degree=3) == (1, 0, 0, 0)


def test_mixed_schemes_are_rejected(bundle_factory):
    """Verify that mixed schemes are rejected."""
    E = bundle_factory()
    F = bundle_factory()

    with pytest.raises(ValueError, match="same scheme"):
        chern_class(E + F)


def test_invalid_component_and_degree_inputs(bundle_factory):
    """Verify that invalid component and degree inputs."""
    E = bundle_factory()

    with pytest.raises(TypeError):
        chern_class(E, component="1")

    with pytest.raises(ValueError):
        chern_class(E, component=-1)

    with pytest.raises(ValueError):
        chern_class(E, max_chern_degree=-1)

    with pytest.raises(ValueError):
        chern_class(E, component=2, max_chern_degree=1)


def test_expression_method_matches_function(explicit_bundles):
    """Verify that expression method matches function."""
    _, E, F, *_ = explicit_bundles
    expression = E + E * F

    assert_tuple_equal(expression.c(), chern_class(expression))
    assert_expr_equal(expression.c(2), chern_class(expression, component=2))


def test_requested_degree_beyond_stored_classes_is_zero_filled(explicit_bundles):
    """Verify that requested degree beyond stored classes is zero filled."""
    _, E, *_ = explicit_bundles
    result = chern_class(E, max_chern_degree=5)

    assert_tuple_equal(result[:4], E.c())
    assert_tuple_equal(result[4:], (0, 0))
