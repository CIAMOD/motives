import pytest
import sympy as sp

from motives.k_theory import DeterminantBundle, DualBundle, Scheme, VectorBundle


def test_scheme_initializes_metadata_and_symbolic_betti_numbers():
    """Verify that scheme initializes metadata and symbolic betti numbers."""
    X = Scheme("X_scheme", 2)
    assert X.name == "X_scheme"
    assert X.dimension == 2
    assert X.max_betti_degree == 4
    assert X.betti_numbers == tuple(sp.Symbol(f"b{i}_X_scheme") for i in range(5))
    assert str(X) == "X_scheme"
    assert repr(X) == "X_scheme"


def test_explicit_betti_numbers_are_sympified():
    """Verify that explicit betti numbers are sympified."""
    X = Scheme("X_betti", sp.Integer(1), [1, "b", 1])
    assert X.dimension == 1
    assert X.betti_numbers == (1, sp.Symbol("b"), 1)


@pytest.mark.parametrize("dimension", [True, False, 1.5, sp.Symbol("d"), "2", None])
def test_dimension_must_be_a_non_boolean_integer(dimension):
    """Verify that dimension must be a non boolean integer."""
    with pytest.raises(TypeError, match="dimension must be an integer"):
        Scheme("X_bad_dimension", dimension)


def test_negative_dimension_is_rejected():
    """Verify that negative dimension is rejected."""
    with pytest.raises(ValueError, match="non-negative"):
        Scheme("X_negative", -1)


@pytest.mark.parametrize("name", ["", None, 3])
def test_name_must_be_nonempty_string(name):
    """Verify that name must be nonempty string."""
    with pytest.raises(ValueError, match="non-empty string"):
        Scheme(name, 1)


def test_betti_number_length_must_equal_two_dimension_plus_one():
    """Verify that betti number length must equal two dimension plus one."""
    with pytest.raises(ValueError, match="exactly 5 components"):
        Scheme("X_wrong_betti", 2, [1, 2, 1])


def test_standard_bundles_have_correct_types_ranks_and_scheme():
    """Verify that standard bundles have correct types ranks and scheme."""
    X = Scheme("X_standard", 3)
    assert isinstance(X.Ox, VectorBundle)
    assert isinstance(X.Tx, VectorBundle)
    assert isinstance(X.Tx_dual, DualBundle)
    assert isinstance(X.Kx, DeterminantBundle)
    assert all(bundle.scheme is X for bundle in (X.Ox, X.Tx, X.Tx_dual, X.Kx))
    assert X.Ox.rank == 1
    assert X.Tx.rank == 3
    assert X.Tx_dual.rank == 3
    assert X.Kx.rank == 1


def test_structure_sheaf_is_trivial_in_all_stored_degrees():
    """Verify that structure sheaf is trivial in all stored degrees."""
    X = Scheme("X_trivial", 3)
    assert X.Ox.c() == (1, 0, 0, 0)
    assert X.Ox.ch() == (1, 0, 0, 0)


def test_cotangent_and_canonical_bundles_are_built_from_tangent_bundle():
    """Verify that cotangent and canonical bundles are built from tangent bundle."""
    X = Scheme("X_geometry", 2)
    assert X.Tx_dual.bundle is X.Tx
    assert X.Kx.bundle is X.Tx_dual
    assert X.Tx_dual.c() == (1, -X.Tx.c(1), X.Tx.c(2))
    assert X.Kx.c() == (1, X.Tx_dual.c(1), 0)


def test_zero_dimensional_scheme_standard_bundles_are_well_defined():
    """Verify that zero dimensional scheme standard bundles are well defined."""
    X = Scheme("Point", 0)
    assert X.betti_numbers == (sp.Symbol("b0_Point"),)
    assert X.Ox.c() == (1,)
    assert X.Tx.rank == 0
    assert X.Tx_dual.c() == (1,)
    assert X.Kx.c() == (1,)
