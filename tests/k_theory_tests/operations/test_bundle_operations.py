import pytest
import sympy as sp

from motives.k_theory import Det, Dual, DualBundle, End, Hom


def test_hom_is_dual_source_tensor_target(bundle_factory):
    """Verify that Hom is dual source tensor target."""
    E = bundle_factory()
    F = bundle_factory(scheme=E.scheme)
    result = Hom(E, F)
    assert result == Dual(E) * F
    dual_atoms = result.atoms(DualBundle)
    assert len(dual_atoms) == 1
    assert next(iter(dual_atoms)).bundle is E


def test_hom_rejects_non_bundle_arguments(bundle_factory):
    """Verify that Hom rejects non bundle arguments."""
    E = bundle_factory()
    with pytest.raises(TypeError, match="Hom expects two VectorBundle objects"):
        Hom(E, sp.Symbol("F"))
    with pytest.raises(TypeError, match="Hom expects two VectorBundle objects"):
        Hom(1, E)


def test_hom_rejects_distinct_scheme_instances(bundle_factory, scheme_factory):
    """Verify that Hom rejects distinct scheme instances."""
    E = bundle_factory(scheme=scheme_factory(name="X_same_name"))
    F = bundle_factory(scheme=scheme_factory(name="X_same_name"))
    with pytest.raises(ValueError, match="same scheme"):
        Hom(E, F)


def test_end_is_hom_from_bundle_to_itself(bundle_factory):
    """Verify that End is Hom from bundle to itself."""
    E = bundle_factory()
    assert End(E) == Hom(E, E) == Dual(E) * E


def test_end_rejects_non_bundle():
    """Verify that End rejects non bundle."""
    with pytest.raises(TypeError, match="End expects a VectorBundle"):
        End(sp.Symbol("E"))


def test_det_and_dual_reject_symbolic_expressions(bundle_factory):
    """Verify that Det and dual reject symbolic expressions."""
    E = bundle_factory()
    F = bundle_factory(scheme=E.scheme)
    with pytest.raises(TypeError):
        Dual(E + F)
    with pytest.raises(TypeError):
        Det(E + F)


def test_double_dual_simplification_does_not_mutate_original(bundle_factory):
    """Verify that double dual simplification does not mutate original."""
    E = bundle_factory()
    original_data = (E.c(), E.ch())
    assert Dual(Dual(E)) is E
    assert (E.c(), E.ch()) == original_data
