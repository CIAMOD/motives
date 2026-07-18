import pytest
import sympy as sp

from motives.k_theory import Curve


def test_curve_has_dimension_one_and_default_symbolic_genus():
    """Verify that curve has dimension one and default symbolic genus."""
    C = Curve("C_symbolic")
    assert C.dimension == 1
    assert C.genus == sp.Symbol("g_C_symbolic")
    assert C.betti_numbers == (1, 2 * C.genus, 1)


def test_curve_with_numeric_genus_has_standard_betti_numbers():
    """Verify that curve with numeric genus has standard betti numbers."""
    C = Curve("C_numeric", genus=4)
    assert C.genus == 4
    assert C.betti_numbers == (1, 8, 1)


def test_curve_accepts_symbolic_genus():
    """Verify that curve accepts symbolic genus."""
    g = sp.Symbol("g", integer=True, nonnegative=True)
    C = Curve("C_g", genus=g)
    assert C.genus == g
    assert C.betti_numbers == (1, 2 * g, 1)


def test_curve_accepts_custom_betti_numbers():
    """Verify that curve accepts custom betti numbers."""
    b1 = sp.Symbol("b1")
    C = Curve("C_custom", genus=7, betti_numbers=[1, b1, 1])
    assert C.betti_numbers == (1, b1, 1)


def test_negative_integer_genus_is_rejected():
    """Verify that negative integer genus is rejected."""
    with pytest.raises(ValueError, match="genus must be non-negative"):
        Curve("C_negative", genus=-1)


def test_custom_betti_numbers_still_require_three_components():
    """Verify that custom betti numbers still require three components."""
    with pytest.raises(ValueError, match="exactly 3 components"):
        Curve("C_bad_betti", genus=2, betti_numbers=[1, 4])


def test_curve_inherits_all_standard_bundles():
    """Verify that curve inherits all standard bundles."""
    C = Curve("C_bundles", genus=2)
    assert C.Ox.rank == 1
    assert C.Tx.rank == 1
    assert C.Tx_dual.rank == 1
    assert C.Kx.rank == 1
    assert all(bundle.max_degree == 1 for bundle in (C.Ox, C.Tx, C.Tx_dual, C.Kx))
