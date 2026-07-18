import pytest
import sympy as sp

from motives.k_theory import Dual, SymPower, Wedge, chern_character

from ..conftest import assert_expr_equal, assert_tuple_equal


def test_bundle_character_is_returned_and_truncated(explicit_bundles):
    """Verify that bundle character is returned and truncated."""
    _, E, *_ = explicit_bundles
    assert chern_character(E) == E.ch()
    assert chern_character(E, max_chern_degree=1) == E.ch()[:2]
    assert chern_character(E, max_chern_degree=5) == E.ch() + (0, 0)


def test_component_selection(explicit_bundles):
    """Verify that component selection."""
    _, E, *_ = explicit_bundles
    assert chern_character(E, component=2) == E.ch(2)
    assert chern_character(E, component=2, max_chern_degree=3) == E.ch(2)


def test_direct_sum_is_componentwise_additive(explicit_bundles):
    """Verify that direct sum is componentwise additive."""
    _, E, F, _, _, eus, fvs = explicit_bundles
    u1, u2, u3 = eus
    v1, v2, v3 = fvs
    assert chern_character(E + F) == (5, u1 + v1, u2 + v2, u3 + v3)


def test_tensor_product_uses_graded_convolution(explicit_bundles):
    """Verify that tensor product uses graded convolution."""
    _, E, F, _, _, eus, fvs = explicit_bundles
    u1, u2, u3 = eus
    v1, v2, v3 = fvs
    expected = (6, 3 * u1 + 2 * v1, 3 * u2 + u1 * v1 + 2 * v2, 3 * u3 + u2 * v1 + u1 * v2 + 2 * v3)
    assert_tuple_equal(chern_character(E * F), expected)


def test_scalar_multiple_scales_character(explicit_bundles):
    """Verify that scalar multiple scales character."""
    _, E, *_ = explicit_bundles
    assert_tuple_equal(chern_character(3 * E), tuple(3 * value for value in E.ch()))


def test_scalar_character_is_concentrated_in_degree_zero():
    """Verify that scalar character is concentrated in degree zero."""
    a = sp.Symbol("a")
    assert chern_character(a, max_chern_degree=3) == (a, 0, 0, 0)


def test_nonnegative_integer_tensor_powers(explicit_bundles):
    """Verify that nonnegative integer tensor powers."""
    _, E, *_ = explicit_bundles
    assert chern_character(E**0, max_chern_degree=3) == (1, 0, 0, 0)
    expected_square = (E.rank**2, 2 * E.rank * E.ch(1), E.ch(1)**2 + 2 * E.rank * E.ch(2), 2 * E.rank * E.ch(3) + 2 * E.ch(1) * E.ch(2))
    assert_tuple_equal(chern_character(E**2), expected_square)


def test_negative_and_noninteger_powers_are_not_supported(bundle_factory):
    """Verify that negative and noninteger powers are not supported."""
    E = bundle_factory()
    with pytest.raises(NotImplementedError, match="non-negative powers"):
        chern_character(E**-1)
    with pytest.raises(NotImplementedError, match="integer powers"):
        chern_character(E**sp.Rational(1, 2))


def test_symmetric_square_character(explicit_bundles):
    """Verify that symmetric square character."""
    _, E, _, _, _, eus, _ = explicit_bundles
    u1, u2, u3 = eus
    r = E.rank
    expected = (r * (r + 1) / 2, (r + 1) * u1, (r + 2) * u2 + u1**2 / 2, (r + 4) * u3 + u1 * u2)
    assert_tuple_equal(chern_character(SymPower(2, E)), expected)


def test_exterior_square_character(explicit_bundles):
    """Verify that exterior square character."""
    _, E, _, _, _, eus, _ = explicit_bundles
    u1, u2, u3 = eus
    r = E.rank
    expected = (r * (r - 1) / 2, (r - 1) * u1, (r - 2) * u2 + u1**2 / 2, (r - 4) * u3 + u1 * u2)
    assert_tuple_equal(chern_character(Wedge(2, E)), expected)


def test_dual_character_alternates_sign(explicit_bundles):
    """Verify that dual character alternates sign."""
    _, E, *_ = explicit_bundles
    assert chern_character(Dual(E)) == tuple((-1)**i * E.ch(i) for i in range(E.max_degree + 1))


def test_mixed_schemes_are_rejected(bundle_factory):
    """Verify that mixed schemes are rejected."""
    E = bundle_factory()
    F = bundle_factory()
    with pytest.raises(ValueError, match="same scheme"):
        chern_character(E + F)


@pytest.mark.parametrize("component", [1.5, "1", None])
def test_invalid_component_types(component, bundle_factory):
    """Verify that invalid component types."""
    if component is None:
        return
    with pytest.raises(TypeError, match="component must be an integer"):
        chern_character(bundle_factory(), component=component)


def test_negative_component_and_degree_are_rejected(bundle_factory):
    """Verify that negative component and degree are rejected."""
    E = bundle_factory()
    with pytest.raises(ValueError, match="component must be non-negative"):
        chern_character(E, component=-1)
    with pytest.raises(ValueError, match="max_chern_degree must be non-negative"):
        chern_character(E, max_chern_degree=-1)
    with pytest.raises(ValueError, match="must be >= component"):
        chern_character(E, component=2, max_chern_degree=1)


def test_expression_method_matches_function(explicit_bundles):
    """Verify that expression method matches function."""
    _, E, F, *_ = explicit_bundles
    expression = E + E * F + Wedge(2, F)
    assert expression.ch() == chern_character(expression)
    assert expression.ch(2) == chern_character(expression, component=2)


def test_line_bundle_exterior_square_vanishes_when_character_is_exponential(scheme_factory):
    """Verify that line bundle exterior square vanishes when character is exponential."""
    from motives.k_theory import VectorBundle
    X = scheme_factory(dimension=3)
    x = sp.Symbol("x")
    L = VectorBundle("L_line", X, rank=1, chern_character=(1, x, x**2 / 2, x**3 / 6))
    assert_tuple_equal(chern_character(Wedge(2, L)), (0, 0, 0, 0))
