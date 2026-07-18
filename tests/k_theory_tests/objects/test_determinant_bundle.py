import pytest
import sympy as sp

from motives.k_theory import Det, DeterminantBundle, VectorBundle


def test_determinant_is_line_bundle_on_same_scheme(explicit_bundles):
    """Verify that determinant is line bundle on same scheme."""
    X, E, *_ = explicit_bundles
    determinant = DeterminantBundle(E)
    assert isinstance(determinant, VectorBundle)
    assert determinant.bundle is E
    assert determinant.scheme is X
    assert determinant.rank == 1
    assert determinant.max_degree == E.max_degree


def test_determinant_chern_classes_have_only_first_class(explicit_bundles):
    """Verify that determinant Chern classes have only first class."""
    _, E, _, cs, *_ = explicit_bundles
    e1, _, _ = cs
    assert DeterminantBundle(E).c() == (1, e1, 0, 0)


def test_determinant_character_is_exponential_of_first_character_component(explicit_bundles):
    """Verify that determinant character is exponential of first character component."""
    _, E, _, _, _, chs, _ = explicit_bundles
    u1, _, _ = chs
    assert DeterminantBundle(E).ch() == (1, u1, u1**2 / 2, u1**3 / 6)


def test_determinant_data_tracks_original_bundle_dynamically(bundle_factory):
    """Verify that determinant data tracks original bundle dynamically."""
    E = bundle_factory(dimension=3)
    determinant = DeterminantBundle(E)
    x, y = sp.symbols("x y")
    E.chern_classes = {1: x}
    E.chern_character = {1: y}
    assert determinant.c() == (1, x, 0, 0)
    assert determinant.ch() == (1, y, y**2 / 2, y**3 / 6)


@pytest.mark.parametrize("attribute", ["chern_classes", "chern_character"])
def test_determinant_characteristic_data_cannot_be_assigned(bundle_factory, attribute):
    """Verify that determinant characteristic data cannot be assigned."""
    determinant = DeterminantBundle(bundle_factory())
    with pytest.raises(AttributeError, match="determined by its original bundle"):
        setattr(determinant, attribute, (1, 0, 0, 0))


def test_determinant_bundle_rejects_non_bundle():
    """Verify that determinant bundle rejects non bundle."""
    with pytest.raises(TypeError, match="DeterminantBundle expects a VectorBundle"):
        DeterminantBundle(sp.Symbol("x"))


def test_det_operation_returns_determinant_bundle(bundle_factory):
    """Verify that Det operation returns determinant bundle."""
    E = bundle_factory()
    determinant = Det(E)
    assert isinstance(determinant, DeterminantBundle)
    assert determinant.bundle is E


def test_det_operation_rejects_non_bundle():
    """Verify that Det operation rejects non bundle."""
    with pytest.raises(TypeError, match="Det expects a VectorBundle"):
        Det(sp.Symbol("x"))


def test_determinant_plain_and_latex_printing(bundle_factory):
    """Verify that determinant plain and LaTeX printing."""
    E = bundle_factory(name="E_det_print")
    determinant = Det(E)
    assert str(determinant) == "Det(E_det_print)"
    assert sp.latex(determinant) == r"\det\left(E_det_print\right)"


def test_zero_dimensional_determinant_has_only_degree_zero(bundle_factory):
    """Verify that zero dimensional determinant has only degree zero."""
    E = bundle_factory(dimension=0, rank=0)
    determinant = Det(E)
    assert determinant.c() == (1,)
    assert determinant.ch() == (1,)
