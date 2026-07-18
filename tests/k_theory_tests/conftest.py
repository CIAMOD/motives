from __future__ import annotations

import itertools

import pytest
import sympy as sp
from sympy.core.cache import clear_cache

from motives.k_theory import Scheme, VectorBundle


_counter = itertools.count()


@pytest.fixture(autouse=True)
def clear_sympy_cache():
    """Prevent SymPy's global symbol cache from leaking mutable bundle metadata."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def scheme_factory():
    """Return a factory producing schemes with globally unique symbolic names."""
    def factory(dimension=3, name=None, betti_numbers=None):
        """Create an object configured by the enclosing fixture."""
        suffix = next(_counter)
        return Scheme(name or f"X_{suffix}", dimension, betti_numbers=betti_numbers)
    return factory


@pytest.fixture
def bundle_factory(scheme_factory):
    """Return a factory producing vector bundles with explicit characteristic data."""
    def factory(name=None, scheme=None, rank=2, dimension=3, chern_classes=None, chern_character=None):
        """Create an object configured by the enclosing fixture."""
        suffix = next(_counter)
        scheme = scheme or scheme_factory(dimension=dimension)
        name = name or f"E_{suffix}"
        return VectorBundle(name, scheme, rank=rank, chern_classes=chern_classes, chern_character=chern_character)
    return factory


@pytest.fixture
def three_bundles(scheme_factory):
    """Return three rank-compatible bundles on one threefold."""
    X = scheme_factory(dimension=3)
    E = VectorBundle(f"E_{next(_counter)}", X, rank=1)
    F = VectorBundle(f"F_{next(_counter)}", X, rank=3)
    G = VectorBundle(f"G_{next(_counter)}", X, rank=2)
    return X, E, F, G


@pytest.fixture
def explicit_bundles(scheme_factory):
    """Return two bundles whose Chern data use compact symbols."""
    X = scheme_factory(dimension=3)
    e1, e2, e3, f1, f2, f3 = sp.symbols("e1 e2 e3 f1 f2 f3")
    u1, u2, u3, v1, v2, v3 = sp.symbols("u1 u2 u3 v1 v2 v3")
    E = VectorBundle(f"E_{next(_counter)}", X, rank=2, chern_classes=(1, e1, e2, e3), chern_character=(2, u1, u2, u3))
    F = VectorBundle(f"F_{next(_counter)}", X, rank=3, chern_classes=(1, f1, f2, f3), chern_character=(3, v1, v2, v3))
    return X, E, F, (e1, e2, e3), (f1, f2, f3), (u1, u2, u3), (v1, v2, v3)


def assert_expr_equal(actual, expected):
    """Assert equality after algebraic simplification."""
    assert sp.simplify(sp.expand(actual - expected)) == 0


def assert_tuple_equal(actual, expected):
    """Assert componentwise equality of symbolic tuples."""
    assert len(actual) == len(expected)
    for actual_component, expected_component in zip(actual, expected):
        assert_expr_equal(actual_component, expected_component)
