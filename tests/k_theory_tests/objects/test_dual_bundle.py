import pytest
import sympy as sp

from motives.k_theory import Dual, DualBundle, VectorBundle


def test_dual_bundle_preserves_scheme_rank_and_degree(explicit_bundles):
    """Verify that dual bundle preserves scheme rank and degree."""
    X, E, *_ = explicit_bundles
    D = DualBundle(E)
    assert isinstance(D, VectorBundle)
    assert D.bundle is E
    assert D.scheme is X
    assert D.rank == E.rank
    assert D.max_degree == E.max_degree


def test_dual_invariants_alternate_sign_by_degree(explicit_bundles):
    """Verify that dual invariants alternate sign by degree."""
    _, E, _, cs, _, chs, _ = explicit_bundles
    e1, e2, e3 = cs
    u1, u2, u3 = chs
    D = DualBundle(E)
    assert D.c() == (1, -e1, e2, -e3)
    assert D.ch() == (E.rank, -u1, u2, -u3)


def test_dual_invariants_are_computed_dynamically(bundle_factory):
    """Verify that dual invariants are computed dynamically."""
    E = bundle_factory(dimension=2)
    D = DualBundle(E)
    x, y = sp.symbols("x y")
    E.chern_classes = {1: x, 2: y}
    E.chern_character = {1: 2 * x, 2: 3 * y}
    assert D.c() == (1, -x, y)
    assert D.ch() == (E.rank, -2 * x, 3 * y)


@pytest.mark.parametrize("attribute", ["chern_classes", "chern_character"])
def test_dual_characteristic_data_cannot_be_assigned_independently(bundle_factory, attribute):
    """Verify that dual characteristic data cannot be assigned independently."""
    D = DualBundle(bundle_factory())
    with pytest.raises(AttributeError, match="determined by its original bundle"):
        setattr(D, attribute, (1, 0, 0, 0))


def test_dual_bundle_rejects_non_vector_bundle():
    """Verify that dual bundle rejects non vector bundle."""
    with pytest.raises(TypeError, match="DualBundle expects a VectorBundle"):
        DualBundle(sp.Symbol("x"))


def test_dual_operation_returns_dual_bundle(bundle_factory):
    """Verify that dual operation returns dual bundle."""
    E = bundle_factory()
    D = Dual(E)
    assert isinstance(D, DualBundle)
    assert D.bundle is E


def test_dual_operation_simplifies_double_dual(bundle_factory):
    """Verify that dual operation simplifies double dual."""
    E = bundle_factory()
    assert Dual(Dual(E)) is E


def test_dual_operation_rejects_non_bundle():
    """Verify that dual operation rejects non bundle."""
    with pytest.raises(TypeError, match="Dual expects a VectorBundle"):
        Dual(1)


def test_dual_plain_and_latex_printing(bundle_factory):
    """Verify that dual plain and LaTeX printing."""
    E = bundle_factory(name="E_print")
    D = Dual(E)
    assert str(D) == "Dual(E_print)"
    assert sp.latex(D) == r"\left(E_print\right)^\vee"


def test_dual_of_zero_dimensional_bundle(bundle_factory):
    """Verify that dual of zero dimensional bundle."""
    E = bundle_factory(dimension=0, rank=5)
    D = Dual(E)
    assert D.c() == (1,)
    assert D.ch() == (5,)
