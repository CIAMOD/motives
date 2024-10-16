from time import perf_counter

from .utils import save_pol
from .core import LambdaRingContext
from .grothendieck_motives.curves.curve import Curve
from .grothendieck_motives.moduli_motives.scheme import TwistedHiggsModuli

def compare_equation(
    g: int, p: int, r: int, filename: str, time: bool = True, *, verbose: int = 0
) -> bool:
    """Check that the motives calculated using both derivations are equal.

    Args:
        g: The genus of the curve.
        p: The number of points on the curve.
        r: The rank of the formula to compare.
        filename: Name of the file where to save the polynomial once computed.
        time: Whether to print the time it took to compute the polynomials.
        verbose: How much to print when deriving the formula. 0 is nothing at all,
            1 is only sentences to see progress, 2 is also intermediate formulas.

    Returns:
        True if the equations are equal, False otherwise.
    """
    start = perf_counter()

    lrc = LambdaRingContext()
    cur = Curve("x", g=g)

    # Compute the motive of rank r using ADHM derivation
    moz = TwistedHiggsModuli(x=cur, p=p, r=r, method="ADHM")
    eq_m = moz.compute(lrc=lrc, verbose=verbose)

    # Compute the motive of rank r using BB derivation
    dav = TwistedHiggsModuli(x=cur, p=p, r=r, method="BB")
    eq_d = dav.compute(lrc=lrc, verbose=verbose)

    if time is True:
        print("Polynomials computed")
        end = perf_counter()
        print(f"------------ Time: {end - start} ------------")

    # Compare the two polynomials
    if eq_d - eq_m == 0:
        save_pol(eq_d, filename)

        if verbose > 0:
            print("Saved polynomial. Success :)")
        if verbose > 1:
            print(f"Polynomial: {eq_d}")

        return True
    return False
