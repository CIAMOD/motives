import sympy as sp
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow

import motives.k_theory as kt
from motives.core.lambda_ring_expr import LambdaRingExpr


def test_root_package_exports_complete_public_api():
    """Verify that root package exports complete public API."""
    expected = {
        "Curve", "Scheme", "VectorBundle", "Wedge", "SymPower", "DualBundle", "DeterminantBundle",
        "Dual", "Hom", "End", "Det", "chern_character", "chern_class", "wedge", "sym", "to_wedge",
        "to_sym", "exact_sequence_realtions", "solve_exact_sequence"
    }
    assert expected <= set(dir(kt))


def test_subpackage_exports_are_importable():
    """Verify that subpackage exports are importable."""
    from motives.k_theory.chern import chern_character, chern_class
    from motives.k_theory.objects import Curve, DeterminantBundle, DualBundle, Scheme, SymPower, VectorBundle, Wedge
    from motives.k_theory.operations import Det, Dual, End, Hom, exact_sequence_realtions, solve_exact_sequence, sym, to_sym, to_wedge, wedge
    assert all(callable(item) for item in (chern_character, chern_class, Curve, DeterminantBundle, DualBundle, Scheme, SymPower, VectorBundle, Wedge, Det, Dual, End, Hom, exact_sequence_realtions, solve_exact_sequence, sym, to_sym, to_wedge, wedge))


def test_expression_types_receive_k_theory_methods_after_import():
    """Verify that expression types receive k theory methods after import."""
    for cls in (LambdaRingExpr, Add, Mul, Pow):
        for method in ("wedge", "sym", "to_wedge", "to_sym", "c", "ch"):
            assert callable(getattr(cls, method))


def test_public_typo_is_stable_until_intentionally_renamed():
    """Verify that public typo is stable until intentionally renamed."""
    assert callable(kt.exact_sequence_realtions)
    assert not hasattr(kt, "exact_sequence_relations")


def test_methods_work_on_add_mul_and_pow(bundle_factory):
    """Verify that methods work on Add Mul and Pow."""
    E = bundle_factory()
    F = bundle_factory(scheme=E.scheme)
    expressions = (E + F, E * F, E**2)
    for expression in expressions:
        assert expression.wedge(0) == 1
        assert expression.sym(1) == expression
        assert isinstance(expression.ch(), tuple)
        assert isinstance(expression.c(), tuple)


def test_scalar_sympy_expression_does_not_gain_bundle_methods_accidentally():
    """Verify that scalar SymPy expression does not gain bundle methods accidentally."""
    x = sp.Symbol("ordinary_symbol")
    assert not hasattr(x, "chern_classes")
