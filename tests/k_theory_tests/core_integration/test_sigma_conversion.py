import sympy as sp

from motives.core.lambda_ring_context import LambdaRingContext
from motives.core.operator.ring_operator import Sigma


def test_free_object_exposes_sigma_variables(bundle_factory):
    """Verify that free object exposes sigma variables."""
    E = bundle_factory()
    assert E.get_sigma_var(0) == 1
    assert E.get_sigma_var(1) == E
    assert E.get_sigma_var(2) == E.sigma(2)
    assert E.get_sigma_var(2, as_symbol=True) == sp.Symbol(f"σ2({E})")


def test_to_sigma_exists_on_bundle_and_composite_expression(bundle_factory):
    """Verify that to sigma exists on bundle and composite expression."""
    E = bundle_factory()
    F = bundle_factory(scheme=E.scheme)
    assert callable(E.to_sigma)
    for expression in (E + F, E * F, E**2):
        assert callable(expression.to_sigma)


def test_sigma_node_converts_to_sigma_polynomial(bundle_factory):
    """Verify that sigma node converts to sigma polynomial."""
    E = bundle_factory()
    expression = E.sigma(2).to_sigma()
    assert sp.expand(expression - E.sigma(2)) == 0


def test_to_sigma_direct_sum_identity(bundle_factory):
    """Verify that to sigma direct sum identity."""
    E = bundle_factory()
    F = bundle_factory(scheme=E.scheme)
    actual = (E + F).sigma(2).to_sigma()
    expected = E.sigma(2) + E * F + F.sigma(2)
    assert sp.expand(actual - expected) == 0


def test_adams_substitution_by_sigma(bundle_factory):
    """Verify that adams substitution by sigma."""
    E = bundle_factory()
    adams_two = E.get_adams_var(2)
    substituted = E._subs_adams_sigma(adams_two, 2)
    context = LambdaRingContext()
    expected = context.get_sigma_2_adams_pol(2).xreplace({context.sigma_vars[1]: E, context.sigma_vars[2]: E.sigma(2)})
    assert sp.expand(substituted - expected) == 0


def test_to_sigma_as_symbol_uses_symbolic_sigma_variables(bundle_factory):
    """Verify that to sigma as symbol uses symbolic sigma variables."""
    E = bundle_factory()
    result = E.sigma(2).to_sigma(as_symbol=True)
    assert result.has(sp.Symbol(f"σ2({E})")) or result == sp.Symbol(f"σ2({E})")


def test_wedge_expansion_relabels_all_sigma_nodes(bundle_factory):
    """Verify that wedge expansion relabels all sigma nodes."""
    E = bundle_factory()
    F = bundle_factory(scheme=E.scheme)
    result = (E + F).to_wedge(3)
    assert not any(type(node) is Sigma for node in sp.preorder_traversal(result))
