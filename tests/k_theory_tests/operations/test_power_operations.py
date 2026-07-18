import pytest
import sympy as sp

from motives.core.operator.ring_operator import Lambda_, Sigma
from motives.k_theory import SymPower, Wedge
from motives.k_theory.operations.power_operations import _replace_lambda_by_sym, _replace_sigma_by_wedge


def test_formal_wedge_boundary_degrees(bundle_factory):
    """Verify that formal wedge boundary degrees."""
    E = bundle_factory()
    assert E.wedge(0) == 1
    assert E.wedge(1) is E
    assert E.wedge(2) == Wedge(2, E)


def test_formal_symmetric_boundary_degrees(bundle_factory):
    """Verify that formal symmetric boundary degrees."""
    E = bundle_factory()
    assert E.sym(0) == 1
    assert E.sym(1) is E
    assert E.sym(2) == SymPower(2, E)


@pytest.mark.parametrize("method", ["wedge", "sym", "to_wedge", "to_sym"])
def test_negative_power_degree_is_rejected(bundle_factory, method):
    """Verify that negative power degree is rejected."""
    E = bundle_factory()
    with pytest.raises(ValueError, match="nonnegative"):
        getattr(E, method)(-1)


def test_to_wedge_without_degree_returns_non_wedge_unchanged(bundle_factory):
    """Verify that to wedge without degree returns non wedge unchanged."""
    E = bundle_factory()
    assert E.to_wedge() is E


def test_to_sym_without_degree_returns_non_symmetric_node_unchanged(bundle_factory):
    """Verify that to sym without degree returns non symmetric node unchanged."""
    E = bundle_factory()
    assert E.to_sym() is E


def test_exterior_square_of_direct_sum(bundle_factory):
    """Verify that exterior square of direct sum."""
    E = bundle_factory()
    F = bundle_factory(scheme=E.scheme)
    actual = (E + F).to_wedge(2)
    expected = Wedge(2, E) + E * F + Wedge(2, F)
    assert sp.expand(actual - expected) == 0


def test_symmetric_square_of_direct_sum(bundle_factory):
    """Verify that symmetric square of direct sum."""
    E = bundle_factory()
    F = bundle_factory(scheme=E.scheme)
    actual = (E + F).to_sym(2)
    expected = SymPower(2, E) + E * F + SymPower(2, F)
    assert sp.expand(actual - expected) == 0


def test_expanded_power_boundary_degrees(bundle_factory):
    """Verify that expanded power boundary degrees."""
    E = bundle_factory()
    assert E.to_wedge(0) == 1
    assert E.to_wedge(1) is E
    assert E.to_sym(0) == 1
    assert E.to_sym(1) is E


def test_formal_nodes_can_expand_themselves(bundle_factory):
    """Verify that formal nodes can expand themselves."""
    E = bundle_factory()
    assert E.wedge(2).to_wedge() == E.to_wedge(2)
    assert E.sym(2).to_sym() == E.to_sym(2)


def test_replacement_helpers_only_replace_base_operator_nodes(bundle_factory):
    """Verify that replacement helpers only replace base operator nodes."""
    E = bundle_factory()
    sigma_expression = Sigma(2, E) + Wedge(3, E)
    lambda_expression = Lambda_(2, E) + SymPower(3, E)
    replaced_sigma = _replace_sigma_by_wedge(sigma_expression)
    replaced_lambda = _replace_lambda_by_sym(lambda_expression)
    assert not any(type(atom) is Sigma for atom in sp.preorder_traversal(replaced_sigma))
    assert not any(type(atom) is Lambda_ for atom in sp.preorder_traversal(replaced_lambda))
    assert Wedge(2, E) in replaced_sigma.args
    assert Wedge(3, E) in replaced_sigma.args
    assert SymPower(2, E) in replaced_lambda.args
    assert SymPower(3, E) in replaced_lambda.args


def test_expansions_do_not_leave_plain_internal_nodes(bundle_factory):
    """Verify that expansions do not leave plain internal nodes."""
    E = bundle_factory()
    F = bundle_factory(scheme=E.scheme)
    exterior = (E + F).to_wedge(3)
    symmetric = (E + F).to_sym(3)
    assert not any(type(node) is Sigma for node in sp.preorder_traversal(exterior))
    assert not any(type(node) is Lambda_ for node in sp.preorder_traversal(symmetric))


def test_methods_are_available_on_composite_expressions(bundle_factory):
    """Verify that methods are available on composite expressions."""
    E = bundle_factory()
    F = bundle_factory(scheme=E.scheme)
    for expression in (E + F, E * F, (E + F)**2):
        assert expression.wedge(2) == Wedge(2, expression)
        assert expression.sym(2) == SymPower(2, expression)
