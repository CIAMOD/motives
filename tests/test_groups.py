import pytest

from motives import GL, PGL, PSL, SL, SO, SP, A, B, C, D, E, F4


@pytest.mark.parametrize("n", range(3, 13))
def test_groups(n: int) -> None:
    pgl = PGL(n)
    pgl_lambda = pgl.to_lambda().expand()
    gl = GL(n)
    gl_lambda = gl.to_lambda().expand()
    psl = PSL(n)
    psl_lambda = psl.to_lambda().expand()
    sl = SL(n)
    sl_lambda = sl.to_lambda().expand()
    so = SO(n)
    so_lambda = so.to_lambda().expand()

    if n % 2 == 0:
        sp = SP(n)
        sp_lambda = sp.to_lambda().expand()

    return
