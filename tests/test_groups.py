import pytest
from motives.grothendieck_motives.groups import SemisimpleG, A,B,C,D,E,F4, G2, GL, SL, PSL, SO, SP


@pytest.mark.parametrize("n", [n for n in range(2,6)])
def test_An_dim(n:int):
    grp=A(n)
    assert A(n).dim==n*(n+2)
    return

@pytest.mark.parametrize("n", [n for n in range(2,6)])
def test_Bn_dim(n:int):
    grp=B(n)
    assert grp.dim==n*(2*n+1)
    return

@pytest.mark.parametrize("n", [n for n in range(2,6)])
def test_Cn_dim(n:int):
    grp=C(n)
    assert grp.dim==n*(2*n+1)
    return

@pytest.mark.parametrize("n", [n for n in range(2,6)])
def test_Dn_dim(n:int):
    grp=D(n)
    assert grp.dim==n*(2*n-1)
    return

def test_E6_dim():
    grp=E(6)
    assert grp.dim==156/2
    return

def test_E7_dim():
    grp=E(7)
    assert grp.dim==266/2
    return

def test_E8_dim():
    grp=E(8)
    assert grp.dim==496/2
    return

def test_F4_dim():
    grp=F4()
    assert grp.dim==104/2
    return

def test_G2_dim():
    grp=G2()
    assert grp.dim==28/2
    return