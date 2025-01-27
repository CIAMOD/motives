import sympy as sp
from sympy.polys.rings import PolyElement
from sympy.ntheory import mobius

from ...curves.curve import Curve

from ...lefschetz import Lefschetz
from ....polynomial_1_var import Polynomial1Var

from ....core.lambda_ring_expr import LambdaRingExpr
from ....utils import (
    all_partitions,
    expand_variable,
    expand_variable_1mtk,
    expand_variable_1mt,
    subs_variable,
    Partitions,
)


class TwistedHiggsModuliADHM:
    """
    The motive of the moduli space of L-twisted Higgs bundles over the curve X.

    This class computes the motive of the moduli space of L-twisted Higgs bundles of rank r
    over the algebraic curve X. It uses the conjectural equation from [Mozgovoy 2012, Conjecture 3]
    as a solution to the ADHM equation, based on [Chuang, Diaconescu, Pan 2011].

    Attributes:
    -----------
    cur : Curve
        The motive of an algebraic curve of genus g >= 2.
    g : int
        The genus of the curve.
    p : int
        A positive integer such that deg(L) = 2g - 2 + p.
    r : int
        The rank of the underlying vector bundles.
    lef : Lefschetz
        The Lefschetz motive.
    t : Polynomial1Var
        The Polynomial1Var t that will be substituted for 1 during calculations.
    T : Polynomial1Var
        The Polynomial1Var T used in the generating function.
    gen_f : LambdaRingExpr
        The generating function used to compute the Mozgovoy formula.
    coeff : dict[sp.Expr, sp.Expr]
        A dictionary storing the coefficients of the generating function.
    """

    def __init__(self, x: Curve, p: int, r: int):
        """
        Initializes the TwistedHiggsModuliADHM class with the given curve and parameters.

        Args:
        -----
        x : Curve
            The curve motive used in the formula.
        p : int
            The number of points of the moduli (deg(L) = 2g - 2 + p).
        r : int
            The rank of the underlying vector bundles.
        """
        self.cur = x
        self.g = self.cur.g
        self.p = p
        self.r = r
        self.lef = Lefschetz()
        self.t, self.T = Polynomial1Var("t"), Polynomial1Var("T")
        self.gen_f: LambdaRingExpr = 0
        self.coeff: dict[sp.Expr, sp.Expr] = {}

    # List of values of the Möbius functions, precomputed for efficiency
    _MOBIUS = [1, -1, -1, 0, -1, 1, -1, 0, 0, 1, -1, 0, -1, 1, 1, 0]

    def _mobius(self, n: int) -> int:
        """
        Computes the Möbius function of n, which is
            - 0 if n is divisible by the square of a prime
            - (-1)^k if n is the product of k different primes

        The values computed with this function are saved for efficiency.

        Args:
        -----
        n : int
            Positive integer.

        Returns:
        --------
        int
            Möbius function of n
        """
        if n > len(self._MOBIUS):
            for i in range(len(self._MOBIUS), n):
                self._MOBIUS[i - 1] = mobius(i)
        return self._MOBIUS[n - 1]

    def gen_func(self) -> LambdaRingExpr:
        """
        Generates the function used to compute the Mozgovoy formula.

        This method constructs the generating function up to rank r using the ADHM equation.

        Returns:
        --------
        LambdaRingExpr
            The generating function up to rank r.
        """
        H_sum: list[LambdaRingExpr] = [0] * (self.r + 1)
        for j in range(1, self.r + 1):
            Hn: LambdaRingExpr = 0
            for partition in all_partitions(j):
                part = Partitions(partition)
                Hp = 1
                for el in part.elements:
                    a = part.a(*el)
                    l = part.l(*el)
                    h = a + l + 1
                    Hp *= (
                        (-(self.t ** (a - l)) * self.lef**a) ** self.p
                        * self.t ** ((1 - self.g) * (2 * l + 1))
                        * self.cur.Z(self.t**h * self.lef**a)
                    )
                Hn += Hp
            H_sum[j] = Hn

        for j in range(1, self.r + 1):
            for k in range(1, self.r + 1):
                pleth = 0
                for h in range(1, self.r // j + 1 - (k - 1)):
                    pleth += H_sum[h] * self.T**h
                if isinstance(pleth, int):
                    continue
                pleth = pleth.adams(j)

                self.gen_f += (
                    sp.Integer((-1) ** (k + 1) * self._mobius(j))
                    / sp.Integer(j * k)
                    * pleth**k
                )

        return self.gen_f

    def moz_lambdas(
        self,
        *,
        verbose: int = 0,
    ) -> PolyElement:
        """
        Computes the motive of rank r as a polynomial of lambda operators.

        This method derives the formula for the motive of rank r and converts it into a
        polynomial of lambda operators using the ADHM conjecture.

        Args:
        -----
        verbose : int, optional
            How much information to print during the computation (0 for none, 1 for progress, 2 for formulas).

        Returns:
        --------
        PolyElement
            The motive of rank r, turned into a polynomial of lambdas.
        """
        if verbose > 0:
            print(
                f"Computing the formula of g={self.g}, p={self.p}, r={self.r} with ADHM"
            )

        if self.gen_f == 0:
            self.gen_func()

        if verbose > 0:
            print("Formula computed.")

        t, T = Polynomial1Var("t"), Polynomial1Var("T")
        lef = Lefschetz()

        if self.coeff == {}:
            sp_M_sum = self.gen_f.to_lambda(as_symbol=True)

            if verbose > 0:
                print("Formula converted to lambdas.")
                print("Max adams degree:", self.gen_f.get_max_adams_degree())
                if verbose > 1:
                    print(sp_M_sum)

            # Prepare the expression for collecting terms with the variable T
            sp_M_sum_expanded = sp_M_sum.replace(*expand_variable(T))
            if verbose > 0:
                print("Formula expanded.")
                if verbose > 1:
                    print(sp_M_sum_expanded)

            self.coeff = sp.collect(sp_M_sum_expanded, T, evaluate=False)
            if verbose > 0:
                print("Coefficients computed.")

        # Multiply terms for easier collection of 1 / (1 - t)
        Hr: sp.Expr = sp.Add(
            *[(1 - t) * (1 - lef * t) * term for term in self.coeff[T**self.r].args]
        )

        # Expand (1 - t ** k) into (1 - t) * (1 + t + ... + t ** (k - 1))
        Hr = Hr.replace(*expand_variable_1mtk(t))

        # Prepare the expression for collection of 1 / (1 - t)
        Hr = Hr.replace(*expand_variable_1mt(t))
        if verbose > 0:
            print("Formula prepared for collection in 1 / (1 - t).")

        poly_domain = sp.ZZ[self.cur.curve_chow.lambda_symbols[1:]]
        domain = sp.ZZ[[lef] + self.cur.curve_chow.lambda_symbols[1:]]

        # Cancel the necessary terms and substitute
        Hr = subs_variable(Hr, t, 1, lef, domain=poly_domain, verbose=verbose)

        et_lambda = (
            domain.from_sympy(
                (-1) ** (self.p * self.r)
                * lef
                ** (self.r**2 * (self.g - 1) + self.p * self.r * (self.r + 1) // 2)
            )
            * Hr
        )

        if verbose > 0:
            print("Final formula computed")
            if verbose > 1:
                print(et_lambda)

        return et_lambda
