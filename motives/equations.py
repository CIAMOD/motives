from time import perf_counter

from .utils import save_pol
from .grothendieck_motives.curves.curve import Curve
from .grothendieck_motives.moduli.scheme import TwistedHiggsModuli


def compare_equation(
    g: int, p: int, r: int, filename: str, time: bool = True, *, verbose: int = 0
) -> bool:
    """Check that the equations for the motive of the moduli space of L-twisted Higgs bundles
    of rank r over a genus g curve calculated using the equation derived from the Bialynicki-Birula
    decomposition of the moduli from [Alfaya, Oliveira 2024, Corollary 8.1] and the conjectural
    equation obtained in [Mozgovoy 2012, Conjecture 3] as a solution to the ADHM equation
    (see [Chuang, Diaconescu, Pan 2011]) are equal.

    Args:
        g: The genus of the curve.
        p: Positive integer such that deg(L)=2g-2+p.
        r: The rank of the the underlying vector bundles. Currently only implemented for r<=3.
        filename: Name of the file where to save the polynomial once computed.
        time: Whether to print the time it took to compute the polynomials.
        verbose: How much to print when deriving the formula. 0 is nothing at all,
            1 is only sentences to see progress, 2 is also intermediate formulas.

    Returns:
        True if the equations are equal, False otherwise.
    """
    start = perf_counter()

    cur = Curve("x", g=g)

    # Compute the motive of rank r using ADHM derivation
    adhm = TwistedHiggsModuli(x=cur, p=p, r=r, method="ADHM")
    eq_adhm = adhm.compute(verbose=verbose)

    # Compute the motive of rank r using BB derivation
    bb = TwistedHiggsModuli(x=cur, p=p, r=r, method="BB")
    eq_bb = bb.compute(verbose=verbose)

    if time is True:
        print("Polynomials computed")
        end = perf_counter()
        print(f"------------ Time: {end - start} ------------")

    # Compare the two polynomials
    if eq_adhm - eq_bb == 0:
        save_pol(eq_bb, filename)

        if verbose > 0:
            print("Saved polynomial. Success.")
        if verbose > 1:
            print(f"Polynomial: {eq_bb}")

        return True
    return False
