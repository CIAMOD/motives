import pytest
import sympy as sp
import math

from motives import Free
from motives.utils import lambda_of_adams_expansion


@pytest.mark.parametrize("n,k", [(i, j) for i in range(2, 6) for j in range(2, 4)])
def test_lambda_of_adams_expansion(n: int, k: int) -> None:
    x = Free("x")
    et = x.adams(k).lambda_(n)
    et = et.to_lambda(as_symbol=False)

    lambda_vars = [x.get_lambda_var(i, as_symbol=False) for i in range(n * k + 1)]
    lambda_expansion = lambda_of_adams_expansion(lambda_vars, n, k)

    result = et - lambda_expansion

    assert result.expand().simplify() == 0
    return


@pytest.mark.parametrize("n,k", [(i, j) for i in range(2, 7) for j in range(2, 5)])
def test_lambda_expansion_2(n: int, k: int) -> None:
    value = 5
    x = Free("x")
    et = lambda_of_adams_expansion(
        [x.get_lambda_var(i, as_symbol=False) for i in range(n * k + 1)], n, k
    )

    lambda_ = lambda n, x: math.comb(n + x - 1, x - 1)
    sigma = lambda n, x: math.comb(x, n)
    adams = lambda n, x: x

    expected = lambda_(n, adams(k, value))

    assert (
        et.subs(
            {
                x.get_lambda_var(i, as_symbol=False): lambda_(i, value)
                for i in range(1, n * k + 1)
            }
        )
        == expected
    )
    return


@pytest.mark.parametrize("n,k", [(i, j) for i in range(2, 7) for j in range(2, 4)])
def test_adams(n: int, k: int) -> None:
    value = 5
    x = sp.Integer(value)
    et = x.adams(k).lambda_(n)

    lambda_ = lambda n, x: math.comb(n + x - 1, x - 1)
    sigma = lambda n, x: math.comb(x, n)
    adams = lambda n, x: x

    expected = lambda_(n, adams(k, value))

    assert et.to_adams(as_symbol=False) == expected
    return


@pytest.mark.parametrize("n,k", [(i, j) for i in range(2, 6) for j in range(2, 4)])
def test_lambda(n: int, k: int) -> None:
    value = 5
    x = Free("x")
    et = x.adams(k).lambda_(n)

    et = et.to_lambda(as_symbol=False)

    lambda_ = lambda n, x: math.comb(n + x - 1, x - 1)
    sigma = lambda n, x: math.comb(x, n)
    adams = lambda n, x: x

    expected = lambda_(n, adams(k, value))

    assert (
        et.xreplace(
            {
                x.get_lambda_var(i, as_symbol=False): lambda_(i, value)
                for i in range(1, n * k + 1)
            }
        )
        == expected
    )
    return


def test_lambda2() -> None:
    value = 5
    x = Free("x")
    et = sum(
        x.lambda_(-i + j + 5) * x.lambda_(1 - i - 2 * j + 5)
        for i in range(1, 5 + 1)
        for j in range(max(-5 + i, 1 - i), math.floor((5 - i + 1) / 2) + 1)
    )

    et_lambda = et.to_lambda(as_symbol=False)

    lambda_ = lambda n, x: math.comb(n + x - 1, x - 1)

    expected = sum(
        lambda_(-i + j + 5, value) * lambda_(1 - i - 2 * j + 5, value)
        for i in range(1, 5 + 1)
        for j in range(max(-5 + i, 1 - i), math.floor((5 - i + 1) / 2) + 1)
    )

    assert (
        et_lambda.xreplace(
            {
                x.get_lambda_var(i, as_symbol=False): lambda_(i, value)
                for i in range(1, 20)
            }
        )
        == expected
    )
    return
