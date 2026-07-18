import pytest
import sympy as sp

from motives.k_theory.chern.chern_class import (
    _chern_character_from_chern_classes, _chern_character_using_chern_classes,
    _chern_classes_from_chern_character, _validate_component_and_degree
)

from ..conftest import assert_tuple_equal


def test_component_and_degree_validation_infers_or_uses_requested_degree(bundle_factory):
    """Verify that component and degree validation infers or uses requested degree."""
    E = bundle_factory(dimension=3)
    assert _validate_component_and_degree(E, None, None) == (None, 3)
    assert _validate_component_and_degree(E, 2, None) == (2, 2)
    assert _validate_component_and_degree(E, 2, 3) == (2, 3)


def test_component_and_degree_validation_errors(bundle_factory):
    """Verify that component and degree validation errors."""
    E = bundle_factory()
    with pytest.raises(TypeError, match="component must be an integer"):
        _validate_component_and_degree(E, 1.5, None)
    with pytest.raises(ValueError, match="component must be non-negative"):
        _validate_component_and_degree(E, -1, None)
    with pytest.raises(ValueError, match="max_chern_degree must be non-negative"):
        _validate_component_and_degree(E, None, -1)
    with pytest.raises(ValueError, match="must be >= component"):
        _validate_component_and_degree(E, 2, 1)


def test_character_from_classes_uses_newton_identities(explicit_bundles):
    """Verify that character from classes uses newton identities."""
    _, E, _, ecs, *_ = explicit_bundles
    c1, c2, c3 = ecs
    expected = (E.rank, c1, c1**2 / 2 - c2, c1**3 / 6 - c1 * c2 / 2 + c3 / 2)
    assert_tuple_equal(_chern_character_from_chern_classes(E, 3), expected)


def test_classes_from_character_use_inverse_newton_identities():
    """Verify that classes from character use inverse newton identities."""
    r, h1, h2, h3 = sp.symbols("r h1 h2 h3")
    expected = (1, h1, h1**2 / 2 - h2, h1**3 / 6 - h1 * h2 + 2 * h3)
    assert_tuple_equal(_chern_classes_from_chern_character((r, h1, h2, h3), 3), expected)


def test_class_character_round_trip(explicit_bundles):
    """Verify that class character round trip."""
    _, E, _, ecs, *_ = explicit_bundles
    character = _chern_character_from_chern_classes(E, 3)
    assert_tuple_equal(_chern_classes_from_chern_character(character, 3), (1,) + ecs)


def test_temporary_character_replacement_is_always_restored(explicit_bundles):
    """Verify that temporary character replacement is always restored."""
    _, E, F, *_ = explicit_bundles
    originals = {E: E.chern_character, F: F.chern_character}
    _chern_character_using_chern_classes(E + F, 3)
    assert E.chern_character == originals[E]
    assert F.chern_character == originals[F]


def test_temporary_character_replacement_is_restored_after_exception(bundle_factory):
    """Verify that temporary character replacement is restored after exception."""
    E = bundle_factory()
    original = E.chern_character
    with pytest.raises(NotImplementedError):
        _chern_character_using_chern_classes(sp.sin(E), E.max_degree)
    assert E.chern_character == original
