import pytest
import sympy as sp

from motives.free import Free
from motives.k_theory import Scheme, VectorBundle


def test_vector_bundle_is_free_symbol_and_preserves_metadata(scheme_factory):
    """Verify that vector bundle is free symbol and preserves metadata."""
    X = scheme_factory(dimension=3)
    E = VectorBundle("E_metadata", X, rank=2)
    assert isinstance(E, Free)
    assert E.name == "E_metadata"
    assert E.scheme is X
    assert E.rank == 2
    assert E.max_degree == 3
    assert str(E) == "E_metadata"


def test_symbolic_rank_is_created_when_rank_is_omitted(scheme_factory):
    """Verify that symbolic rank is created when rank is omitted."""
    X = scheme_factory(dimension=2)
    E = VectorBundle("E_symbolic_rank", X)
    assert E.rank == sp.Symbol("rk_E_symbolic_rank")
    assert E.ch(0) == E.rank


def test_scheme_dimension_overrides_explicit_max_degree(scheme_factory):
    """Verify that scheme dimension overrides explicit max degree."""
    X = scheme_factory(dimension=2)
    E = VectorBundle("E_dimension", X, rank=7, max_degree=9)
    assert E.max_degree == 2
    assert len(E.c()) == 3
    assert len(E.ch()) == 3


def test_default_chern_data_uses_expected_symbols(scheme_factory):
    """Verify that default Chern data uses expected symbols."""
    X = scheme_factory(dimension=3)
    E = VectorBundle("E_defaults", X, rank=4)
    assert E.c() == (1, sp.Symbol("c1_E_defaults"), sp.Symbol("c2_E_defaults"), sp.Symbol("c3_E_defaults"))
    assert E.ch() == (4, sp.Symbol("ch1_E_defaults"), sp.Symbol("ch2_E_defaults"), sp.Symbol("ch3_E_defaults"))


def test_list_and_tuple_inputs_are_sympified_and_zero_padded(scheme_factory):
    """Verify that list and tuple inputs are sympified and zero padded."""
    X = scheme_factory(dimension=3)
    E = VectorBundle("E_padding", X, rank=2, chern_classes=[1, "x"], chern_character=(2, "y", 3))
    assert E.c() == (1, sp.Symbol("x"), 0, 0)
    assert E.ch() == (2, sp.Symbol("y"), 3, 0)


def test_dictionary_setter_updates_only_selected_components(bundle_factory):
    """Verify that dictionary setter updates only selected components."""
    E = bundle_factory(dimension=3)
    original_c = E.c()
    original_ch = E.ch()
    x, y = sp.symbols("x y")
    E.chern_classes = {2: x}
    E.chern_character = {1: y}
    assert E.c(2) == x
    assert E.ch(1) == y
    assert E.c(1) == original_c[1]
    assert E.ch(2) == original_ch[2]


def test_none_restores_default_symbolic_data(bundle_factory):
    """Verify that none restores default symbolic data."""
    E = bundle_factory(dimension=2, chern_classes=(1, 7, 8), chern_character=(2, 9, 10))
    E.chern_classes = None
    E.chern_character = None
    assert E.c() == (1, sp.Symbol(f"c1_{E.name}"), sp.Symbol(f"c2_{E.name}"))
    assert E.ch() == (E.rank, sp.Symbol(f"ch1_{E.name}"), sp.Symbol(f"ch2_{E.name}"))


@pytest.mark.parametrize("attribute", ["chern_classes", "chern_character"])
def test_too_many_components_raise_value_error(bundle_factory, attribute):
    """Verify that too many components raise value error."""
    E = bundle_factory(dimension=2)
    with pytest.raises(ValueError, match="Expected at most 3 components"):
        setattr(E, attribute, [0, 1, 2, 3])


@pytest.mark.parametrize("attribute", ["chern_classes", "chern_character"])
@pytest.mark.parametrize("index", [-1, 3])
def test_dictionary_index_outside_range_raises_index_error(bundle_factory, attribute, index):
    """Verify that dictionary index outside range raises index error."""
    E = bundle_factory(dimension=2)
    with pytest.raises(IndexError, match="out of range"):
        setattr(E, attribute, {index: 0})


@pytest.mark.parametrize("attribute", ["chern_classes", "chern_character"])
def test_unsupported_setter_input_raises_type_error(bundle_factory, attribute):
    """Verify that unsupported setter input raises type error."""
    E = bundle_factory(dimension=2)
    with pytest.raises(TypeError, match="Expected None"):
        setattr(E, attribute, "invalid")


def test_component_selectors_support_none_scalar_list_and_tuple(bundle_factory):
    """Verify that component selectors support none scalar list and tuple."""
    E = bundle_factory(dimension=3)
    assert E.c() == E.chern_classes
    assert E.ch() == E.chern_character
    assert E.c(1) == E.chern_classes[1]
    assert E.ch(2) == E.chern_character[2]
    assert E.c([1, 3]) == (E.c(1), E.c(3))
    assert E.ch((0, 2)) == (E.ch(0), E.ch(2))


@pytest.mark.parametrize("selector", [4, -5, [1, 4]])
def test_component_selector_propagates_index_error(bundle_factory, selector):
    """Verify that component selector propagates index error."""
    E = bundle_factory(dimension=3)
    with pytest.raises(IndexError):
        E.c(selector)


def test_characteristic_data_can_be_updated_after_construction(bundle_factory):
    """Verify that characteristic data can be updated after construction."""
    E = bundle_factory(dimension=2)
    E.chern_classes = (1, 2, 3)
    E.chern_character = (5, 7, 11)
    assert E.c() == (1, 2, 3)
    assert E.ch() == (5, 7, 11)


def test_bundle_participates_in_sympy_expressions(bundle_factory):
    """Verify that bundle participates in SymPy expressions."""
    E = bundle_factory()
    F = bundle_factory(scheme=E.scheme)
    expression = 2 * E + E * F + F**2
    assert expression.has(E)
    assert expression.has(F)
    assert E in expression.atoms(VectorBundle)
    assert F in expression.atoms(VectorBundle)


def test_zero_dimensional_bundle_stores_only_degree_zero(scheme_factory):
    """Verify that zero dimensional bundle stores only degree zero."""
    X = Scheme("Point_for_bundle", 0)
    E = VectorBundle("E_point", X, rank=3)
    assert E.max_degree == 0
    assert E.c() == (1,)
    assert E.ch() == (3,)
