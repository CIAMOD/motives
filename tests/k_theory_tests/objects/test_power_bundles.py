import sympy as sp

from motives.core.operator.ring_operator import Lambda_, Sigma
from motives.k_theory import SymPower, Wedge


def test_wedge_is_sigma_subclass_with_degree_and_child(bundle_factory):
    """Verify that wedge is sigma subclass with degree and child."""
    E = bundle_factory()
    operation = Wedge(2, E)
    assert isinstance(operation, Sigma)
    assert operation.degree == 2
    assert operation.child == E
    assert operation.args == (2, E)


def test_symmetric_power_is_lambda_subclass_with_degree_and_child(bundle_factory):
    """Verify that symmetric power is lambda subclass with degree and child."""
    E = bundle_factory()
    operation = SymPower(3, E)
    assert isinstance(operation, Lambda_)
    assert operation.degree == 3
    assert operation.child == E
    assert operation.args == (3, E)


def test_wedge_printing(bundle_factory):
    """Verify that wedge printing."""
    E = bundle_factory(name="E_wedge")
    operation = Wedge(2, E)
    assert str(operation) == "∧2(E_wedge)"
    assert sp.latex(operation) == r"\wedge^{2}\left(E_wedge\right)"


def test_symmetric_power_printing(bundle_factory):
    """Verify that symmetric power printing."""
    E = bundle_factory(name="E_sym")
    operation = SymPower(2, E)
    assert str(operation) == "Sym2(E_sym)"
    assert sp.latex(operation) == r"\operatorname{Sym}^{2}\left(E_sym\right)"


def test_power_nodes_are_sympy_expressions(bundle_factory):
    """Verify that power nodes are SymPy expressions."""
    E = bundle_factory()
    assert isinstance(Wedge(2, E), sp.Expr)
    assert isinstance(SymPower(2, E), sp.Expr)
    assert Wedge(2, E).has(E)
    assert SymPower(2, E).has(E)
