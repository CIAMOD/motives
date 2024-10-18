import pytest

from motives import GL, LambdaRingExpr

@pytest.mark.parametrize("k,n", [(k, n) for k in range(1, 6) for n in range(1, 15)])
def test_lambda(k: int, n: int) -> None:
    et: LambdaRingExpr = GL(n).lambda_(k)

    lambda_opt = et.to_lambda()
    lambda_wo_opt = et.to_lambda(optimize=False)

    assert (lambda_opt - lambda_wo_opt).simplify() == 0
    return

@pytest.mark.parametrize("k,n", [(k, n) for k in range(1, 6) for n in range(1, 15)])
def test_adams(k: int, n: int) -> None:
    proj = GL(n)

    et: LambdaRingExpr = proj.adams(k)

    et_adams = et.to_adams()
    et_adams_comp = proj.get_adams_var(k)

    assert (et_adams - et_adams_comp).simplify() == 0
    return
