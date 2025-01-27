import pytest

from motives import Jacobian, LambdaRingExpr, Curve, Lefschetz


@pytest.mark.parametrize("k,n", [(k, n) for k in range(1, 6) for n in range(1, 5)])
def test_lambda(k: int, n: int) -> None:
    cur = Curve("C", n)

    et: LambdaRingExpr = Jacobian(cur).lambda_(k)

    lambda_opt = et.to_lambda(as_symbol=False)
    lambda_wo_opt = et.to_lambda(as_symbol=False, optimize=False)

    assert (lambda_opt - lambda_wo_opt).simplify() == 0
    return


@pytest.mark.parametrize("k,n", [(k, n) for k in range(1, 6) for n in range(1, 15)])
def test_adams(k: int, n: int) -> None:
    cur = Curve("C", n)

    jac = Jacobian(cur)

    et: LambdaRingExpr = jac.adams(k)

    et_adams = et.to_adams(as_symbol=False)
    et_adams_comp = jac.get_adams_var(k, as_symbol=False)

    assert (et_adams - et_adams_comp).simplify() == 0
    return


def test() -> None:
    cur1 = Curve("C", 2)

    jac1 = Jacobian(cur1)

    et1: LambdaRingExpr = jac1.adams(1)

    et_adams1 = et1.to_adams(as_symbol=False)
    et_adams_comp1 = jac1.get_adams_var(1, as_symbol=False)

    cur2 = Curve("C", 2)

    jac2 = Jacobian(cur2)

    et2: LambdaRingExpr = jac2.adams(2)

    et_adams2 = et2.to_adams(as_symbol=False)
    et_adams_comp2 = jac2.get_adams_var(2, as_symbol=False)

    assert (et_adams2 - et_adams_comp2).simplify() == 0
    assert (et_adams1 - et_adams_comp1).simplify() == 0
    return
