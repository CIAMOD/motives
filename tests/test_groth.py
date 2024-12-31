import pytest
import sympy as sp

from motives.core import LambdaRingContext



def test_generate_adams_2_lambda_pols() -> None:
    N=6
    adams_2_lambda_pols_partitions=N*[0]
    adams_2_lambda_pols_recurrent=N*[0]
    context=LambdaRingContext()
    context._generate_variables(N)
    context.mode="partitions"
    context._compute_adams_2_lambda_pols_partitions(N)
    for i in range(N):
        adams_2_lambda_pols_partitions[i]=context.get_adams_2_lambda_pol(i+1)
    context.mode="recurrent"
    context._compute_adams_2_lambda_pols_recurrent(N)
    context._adams_2_lambda_pols=[]
    for i in range(N):
        adams_2_lambda_pols_recurrent[i]=context.get_adams_2_lambda_pol(i+1)

    for i in range(N):
        assert (adams_2_lambda_pols_partitions[i]-adams_2_lambda_pols_recurrent[i]).simplify()==0
    return




def test_generate_lambda_2_adams_pols() -> None:
    N=6
    lambda_2_adams_pols_partitions=N*[0]
    lambda_2_adams_pols_recurrent=N*[0]
    context=LambdaRingContext()
    context._generate_variables(N)
    context.mode="partitions"
    context._compute_lambda_2_adams_pols_partitions(N)
    for i in range(N):
        lambda_2_adams_pols_partitions[i]=context.get_lambda_2_adams_pol(i+1)
    context.mode="recurrent"
    context._compute_adams_2_lambda_pols_recurrent(N)
    context._lambda_2_adams_pols=[]
    for i in range(N):
        lambda_2_adams_pols_recurrent[i]=context.get_lambda_2_adams_pol(i+1)

    for i in range(N):
        assert (lambda_2_adams_pols_partitions[i]-lambda_2_adams_pols_recurrent[i]).simplify()==0
    return