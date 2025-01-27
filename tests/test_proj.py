import pytest

from motives import Proj, LambdaRingExpr


@pytest.mark.parametrize("k,n", [(k, n) for k in range(1, 6) for n in range(1, 15)])
def test_lambda(k: int, n: int) -> None:
    et: LambdaRingExpr = Proj(n).lambda_(k)

    lambda_opt = et.to_lambda(as_symbol=False)
    lambda_wo_opt = et.to_lambda(as_symbol=False, optimize=False)

    assert (lambda_opt - lambda_wo_opt).simplify() == 0
    return


@pytest.mark.parametrize("k,n", [(k, n) for k in range(1, 6) for n in range(1, 15)])
def test_adams(k: int, n: int) -> None:
    proj = Proj(n)

    et = proj.adams(k)

    et_adams = et.to_adams(as_symbol=False)
    et_adams_comp = proj.get_adams_var(k, as_symbol=False)

    assert (et_adams - et_adams_comp).simplify() == 0
    return
