import pytest
import sympy as sp

from motives.k_theory import SymPower, Wedge
from motives.k_theory.chern.chern_character import (
    _adams_chern, _bundle_max_chern_degree, _ch_add, _ch_dual, _ch_mul, _ch_scalar_mul,
    _ch_sym_power, _ch_wedge_power, _component, _compute_chern_character, _infer_max_chern_degree,
    _is_scalar_expression, _one_ch, _validate_same_scheme, _vector_bundles, _zero_ch
)


def test_component_returns_zero_outside_stored_range():
    """Verify that component returns zero outside stored range."""
    values = (1, 2)
    assert _component(values, 0) == 1
    assert _component(values, 1) == 2
    assert _component(values, -1) == 0
    assert _component(values, 2) == 0


def test_zero_and_one_characters():
    """Verify that zero and one characters."""
    assert _zero_ch(3) == (0, 0, 0, 0)
    assert _one_ch(3) == (1, 0, 0, 0)


def test_character_addition_scaling_dual_and_product():
    """Verify that character addition scaling dual and product."""
    A = (2, 3, 5)
    B = (7, 11, 13)
    assert _ch_add(A, B, 2) == (9, 14, 18)
    assert _ch_scalar_mul(A, 2, 2) == (4, 6, 10)
    assert _ch_dual(A, 2) == (2, -3, 5)
    assert _ch_mul(A, B, 2) == (14, 43, 94)


def test_character_helpers_zero_fill_shorter_tuples():
    """Verify that character helpers zero fill shorter tuples."""
    assert _ch_add((1,), (2, 3), 2) == (3, 3, 0)
    assert _ch_mul((1,), (2, 3), 2) == (2, 3, 0)


def test_adams_operation_scales_degree_i_by_k_to_i():
    """Verify that adams operation scales degree i by k to i."""
    assert _adams_chern((2, 3, 5, 7), 3, 3) == (2, 9, 45, 189)


def test_adams_operation_requires_positive_degree():
    """Verify that adams operation requires positive degree."""
    with pytest.raises(ValueError, match="k >= 1"):
        _adams_chern((1, 2), 0, 1)


def test_symmetric_and_exterior_recurrence_rank_components():
    """Verify that symmetric and exterior recurrence rank components."""
    r = sp.Symbol("r")
    A = (r, sp.Symbol("a1"), sp.Symbol("a2"))
    assert sp.simplify(_ch_sym_power(A, 2, 2)[0] - r * (r + 1) / 2) == 0
    assert sp.simplify(_ch_wedge_power(A, 2, 2)[0] - r * (r - 1) / 2) == 0


def test_symmetric_and_exterior_power_zero_are_multiplicative_identity():
    """Verify that symmetric and exterior power zero are multiplicative identity."""
    A = (2, 3, 5)
    assert _ch_sym_power(A, 0, 2) == (1, 0, 0)
    assert _ch_wedge_power(A, 0, 2) == (1, 0, 0)


def test_negative_power_helpers_raise_value_error():
    """Verify that negative power helpers raise value error."""
    with pytest.raises(ValueError, match="n >= 0"):
        _ch_sym_power((1,), -1, 0)
    with pytest.raises(ValueError, match="n >= 0"):
        _ch_wedge_power((1,), -1, 0)


def test_vector_bundle_atom_discovery(bundle_factory):
    """Verify that vector bundle atom discovery."""
    E = bundle_factory()
    F = bundle_factory(scheme=E.scheme)
    assert _vector_bundles(E) == (E,)
    assert set(_vector_bundles(E + F)) == {E, F}
    assert _vector_bundles(sp.Symbol("x")) == ()


def test_scheme_validation_accepts_one_scheme_and_rejects_mixed_schemes(bundle_factory):
    """Verify that scheme validation accepts one scheme and rejects mixed schemes."""
    E = bundle_factory()
    F = bundle_factory(scheme=E.scheme)
    G = bundle_factory()
    _validate_same_scheme(E + F)
    with pytest.raises(ValueError, match="same scheme"):
        _validate_same_scheme(E + G)


def test_degree_inference_uses_minimum_bundle_degree(bundle_factory):
    """Verify that degree inference uses minimum bundle degree."""
    E = bundle_factory(dimension=3)
    F = bundle_factory(dimension=2)
    assert _bundle_max_chern_degree(E) == 3
    assert _infer_max_chern_degree(E) == 3
    assert _infer_max_chern_degree(E + F) == 2
    assert _infer_max_chern_degree(sp.Symbol("x")) == 5


def test_scalar_expression_detection(bundle_factory):
    """Verify that scalar expression detection."""
    E = bundle_factory()
    x = sp.Symbol("x")
    assert _is_scalar_expression(x + 2)
    assert not _is_scalar_expression(x + E)


def test_internal_computation_dispatches_all_supported_node_types(explicit_bundles):
    """Verify that internal computation dispatches all supported node types."""
    _, E, F, *_ = explicit_bundles
    assert _compute_chern_character(E, 2) == E.ch()[:3]
    assert _compute_chern_character(E + F, 2)[0] == E.rank + F.rank
    assert _compute_chern_character(E * F, 2)[0] == E.rank * F.rank
    assert _compute_chern_character(E**2, 2)[0] == E.rank**2
    assert _compute_chern_character(SymPower(2, E), 2)[0] == E.rank * (E.rank + 1) / 2
    assert _compute_chern_character(Wedge(2, E), 2)[0] == E.rank * (E.rank - 1) / 2
    assert _compute_chern_character(sp.Symbol("a"), 2) == (sp.Symbol("a"), 0, 0)


def test_internal_computation_rejects_unsupported_nodes(bundle_factory):
    """Verify that internal computation rejects unsupported nodes."""
    E = bundle_factory()
    with pytest.raises(NotImplementedError, match="not supported"):
        _compute_chern_character(sp.sin(E), E.max_degree)
