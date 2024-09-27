from src.expression_tree import ET, Curve, Lefschetz, Integer, Symbol, Add
from src.utils import *
from src.grothGroupContext import GrothGroupContext

import math
import sympy as sp
from sympy import QQ, ZZ
from time import perf_counter


class DavidR:
    def __init__(self, g: int, p: int, r: int):
        self.g = g
        self.p = p
        self.r = r
        self.dl = 2 * g - 2 + p

        self.lef = Lefschetz()
        self.cur = Curve(sp.Symbol("C"), g=g)

        # dictionary that stores the different vhs components of the formula
        self.vhs: dict[tuple, ET] = {}

        if self.r == 2:
            self.M2()
        elif self.r == 3:
            self.M3()
        else:
            raise ValueError("The rank should be either 2 or 3")

    def M2(self) -> None:
        """
        Computes the Mozgovoy formula of rank 2, using David's derivation.

        Returns
        -------
        None
        """
        self.vhs[(2,)] = (
            self.lef ** (4 * self.dl + 4 - 4 * self.g)
            * (self.cur.Jac * self.cur.P(self.lef) - self.lef**self.g * self.cur.Jac**2)
            / ((self.lef - 1) * (self.lef**2 - 1))
        )  # VHS 2
        self.vhs[(1, 1)] = (
            self.lef ** (3 * self.dl + 2 - 2 * self.g)
            * self.cur.Jac
            * Add(
                [
                    self.cur.lambda_(1 - 2 * i + self.dl)
                    for i in range(1, (self.dl + 1) // 2 + 1)
                ]
            )
        )  # VHS 1, 1

    def M3(self) -> None:
        """
        Computes the Mozgovoy formula of rank 3, using David's derivation.

        Returns
        -------
        None
        """
        self.vhs[(3,)] = (
            self.lef ** (9 * self.dl + 9 - 9 * self.g)
            * self.cur.Jac
            * (
                self.lef ** (3 * self.g - 1)
                * (1 + self.lef + self.lef**2)
                * self.cur.Jac**2
                - self.lef ** (2 * self.g - 1)
                * (1 + self.lef) ** 2
                * self.cur.Jac
                * self.cur.P(self.lef)
                + self.cur.P(self.lef) * self.cur.P(self.lef**2)
            )  # Vector bundle 3
            / ((self.lef - 1) * (self.lef**2 - 1) ** 2 * (self.lef**3 - 1))
        )
        self.vhs[(1, 2)] = (
            (self.lef ** (7 * self.dl + 5 - 5 * self.g) * self.cur.Jac**2)
            * Add(
                [
                    self.lef ** (i + self.g)
                    # * (self.cur + self.lef**2).lambda_(-2 * i + self.dl)
                    * Add(
                        [
                            self.cur.lambda_(j)
                            # * (self.lef**2).lambda_(-2 * i + self.dl - j)
                            * self.lef ** (2 * (-2 * i + self.dl - j))
                            for j in range(-2 * i + self.dl + 1)
                        ]
                    )
                    # - (self.cur * self.lef + 1).lambda_(-2 * i + self.dl)
                    - Add(
                        [  # (self.cur * self.lef).lambda_(j)
                            self.cur.lambda_(j) * self.lef**j
                            for j in range(-2 * i + self.dl + 1)
                        ]
                    )
                    for i in range(1, math.floor(self.dl / 2 + 1 / 3) + 1)
                ]
            )  # VHS 1, 2
            / (self.lef - 1)
        )
        self.vhs[(2, 1)] = (
            (self.lef ** (7 * self.dl + 5 - 5 * self.g) * self.cur.Jac**2)
            * Add(
                [
                    self.lef ** (i + self.g - 1)
                    # * (self.cur + self.lef**2).lambda_(-2 * i + self.dl + 1)
                    * Add(
                        [
                            self.cur.lambda_(j)
                            # * (self.lef**2).lambda_(-2 * i + self.dl + 1 - j)
                            * self.lef ** (2 * (-2 * i + self.dl + 1 - j))
                            for j in range(-2 * i + self.dl + 1 + 1)
                        ]
                    )
                    # - (self.cur * self.lef + 1).lambda_(-2 * i + self.dl + 1)
                    - Add(
                        [  # (self.cur * self.lef).lambda_(j)
                            self.cur.lambda_(j) * self.lef**j
                            for j in range(-2 * i + self.dl + 1 + 1)
                        ]
                    )
                    for i in range(1, math.floor(self.dl / 2 + 2 / 3) + 1)
                ]
            )  # VHS 2, 1
            / (self.lef - 1)
        )
        self.vhs[(1, 1, 1)] = (
            self.lef ** (6 * self.dl + 3 - 3 * self.g)
            * self.cur.Jac
            * Add(
                [
                    self.cur.lambda_(-i + j + self.dl)
                    * self.cur.lambda_(1 - i - 2 * j + self.dl)
                    for i in range(1, self.dl + 1)
                    for j in range(
                        max(-self.dl + i, 1 - i), math.floor((self.dl - i + 1) / 2) + 1
                    )
                ]
            )  # VHS 1, 1, 1
        )

    def simplify(
        self, group_context: GrothGroupContext = None, *, verbose: int = 0
    ) -> PolyElement:
        """
        Transforms the mozgovoy equation to lambda's and cancels it
        """
        if verbose > 0:
            print("Simplifying Mozgovoy's equation using David's derivation")

        result: sp.Poly = 0
        domain = QQ[[self.lef.L_VAR] + self.cur.curve_hodge.lambda_symbols[1:]]

        # Initialize a GrothGroupContext that will be reused for all the vhs
        if group_context is None:
            group_context = GrothGroupContext()

        if verbose > 0:
            print("Grothendieck group context computed")

        # Add the adams variables to all the variables in the operands dictionary
        for vhs, expr in self.vhs.items():
            # Convert to lambdas
            lambda_pol = expr.to_lambda(group_context=group_context)

            if verbose > 0:
                print(f"Computed lambda of VHS {vhs}")
            if verbose > 1:
                print(f"Lambda of VHS {vhs}: {lambda_pol}")

            if all(el == 1 for el in vhs):
                result += domain.from_sympy(lambda_pol)
                if verbose > 0:
                    print(f"Cancelled VHS {vhs}")
                if verbose > 1:
                    print(f"Cancelled VHS {vhs}: {numerator_pol}")
                continue

            # Cancel the denominator
            numerator, denominator = sp.fraction(lambda_pol)

            numerator_pol = domain.from_sympy(numerator)
            denominator_pol = domain.from_sympy(denominator)

            numerator_pol = numerator_pol.exquo(denominator_pol)

            result += numerator_pol

            if verbose > 0:
                print(f"Cancelled VHS {vhs}")
            if verbose > 1:
                print(f"Cancelled VHS {vhs}: {numerator_pol}")

        return result


class MozR:
    """ """

    def __init__(self, g: int, p: int, r_max: int):
        self.g = g
        self.p = p
        self.r_max = r_max
        self.mobius = [1, -1, -1, 0, -1, 1, -1, 0, 0, 1, -1, 0, -1, 1, 1, 0]

        self.lef = Lefschetz()
        self.cur = Curve(sp.Symbol("C"), g=g)
        self.t, self.T = Symbol(sp.Symbol("t")), Symbol(sp.Symbol("T"))

        self.gen_f: ET = 0
        self.coeff: dict[sp.Expr, sp.Expr] = {}

    def gen_func(self) -> ET:
        """
        The generating function used to compute the Mozgovoy formula,
        using Mozgovoy's derivation.

        Returns
        -------
        ET
            The generating function up to rank r_max.
        """
        H_sum: list[ET] = [0] * (self.r_max + 1)
        for j in range(1, self.r_max + 1):

            Hn: ET = 0
            for partition in generate_partitions(j):
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

        for j in range(1, self.r_max + 1):
            for k in range(1, self.r_max + 1):

                pleth = 0
                for h in range(1, self.r_max // j + 1 - (k - 1)):
                    pleth += H_sum[h] * self.T**h
                if isinstance(pleth, int):
                    continue
                pleth = pleth.adams(j)

                self.gen_f += (
                    Integer((-1) ** (k + 1) * self.mobius[j - 1])
                    / Integer(j * k)
                    * pleth**k
                )

        return self.gen_f

    def moz_lambdas(
        self,
        r: int,
        group_context: GrothGroupContext = None,
        *,
        verbose: int = 0,
    ) -> PolyElement:
        """
        The Mozgovoy formula of rank r given as a polynomial of lambdas,
        using Mozgovoy's derivation.

        Parameters
        ----------
        r : int
            The rank of the formula to return.
        verbose : int
            How much to print when deriving the formula. 0 is nothing at all,
            1 is only sentences to see progress, 2 is also intermediate formulas.

        Returns
        -------
        ET
            The Mozgovoy formula of rank r turned into a polynomial of lambdas.
        """

        if verbose > 0:
            print("Computing the mozgovoy formula using Mozgovoy's derivation.")

        if self.gen_f == 0:
            self.gen_func()

        if verbose > 0:
            print("Formula computed.")

        t, T = sp.Symbol("t"), sp.Symbol("T")
        lef = Lefschetz.L_VAR

        if self.coeff == {}:

            if group_context is None:
                group_context = GrothGroupContext()
            sp_M_sum = self.gen_f.to_lambda(group_context=group_context)

            if verbose > 0:
                print("Formula converted to lambdas.")
                print("Max adams degree:", self.gen_f.get_max_adams_degree())
                if verbose > 1:
                    print(sp_M_sum)

            # Prepare the expression for collection for the variable T
            sp_M_sum_expanded = sp_M_sum.replace(*expand_variable(T))
            if verbose > 0:
                print("Formula expanded.")
                if verbose > 1:
                    print(sp_M_sum_expanded)

            self.coeff = sp.collect(sp_M_sum_expanded, T, evaluate=False)
            if verbose > 0:
                print("Coefficients computed.")

        # Multiply it in this way (distribution of the product) to make it
        # easier to collect 1 / (1 - t) later
        Hr: sp.Expr = sp.Add(
            *[(1 - t) * (1 - lef * t) * term for term in self.coeff[T**r].args]
        )

        # Expand (1 - t ** k) into (1 - t) * (1 + t + ... + t ** (k - 1))
        Hr = Hr.replace(*expand_variable_1mtk(t))

        # Prepare the expression for collection for 1 / (1 - t)
        Hr = Hr.replace(*expand_variable_1mt(t))
        if verbose > 0:
            print("Formula prepared for collection in 1 / (1 - t).")

        poly_domain = ZZ[self.cur.curve_hodge.lambda_symbols[1:]]
        domain = ZZ[[lef] + self.cur.curve_hodge.lambda_symbols[1:]]

        # Cancel the necessary terms to avoid indeterminate forms and substitute
        Hr = subs_variable(Hr, t, 1, lef, domain=poly_domain, verbose=verbose)

        et_lambda = (
            domain.from_sympy(
                (-1) ** (self.p * r)
                * lef ** (r**2 * (self.g - 1) + self.p * r * (r + 1) // 2)
            )
            * Hr
        )

        if verbose > 0:
            print("Final formula computed")
            if verbose > 1:
                print(et_lambda)

        return et_lambda


def compare_equation(
    g: int, p: int, r: int, filename: str, time: bool = True, *, verbose: int = 0
) -> bool:
    """
    Compare the Mozgovoy formula of rank r with the David's derivation
    of the Mozgovoy formula of rank r.

    Parameters
    ----------
    g : int
        The genus of the curve.
    p : int
        The number of points on the curve.
    r : int
        The rank of the formula to compare.
    filename: str
        Name of the file where to save the polynomial once computed

    Returns
    -------
    bool
        True if the equations are equal, False otherwise.
    """
    start = perf_counter()

    group_context = GrothGroupContext()

    # Compute the Mozgovoy formula of rank r using Mozgovoy's derivation
    moz = MozR(g=g, p=p, r_max=r)
    eq_m_zz = moz.moz_lambdas(r=r, group_context=group_context, verbose=verbose)

    # Compute the Mozgovoy formula of rank r using David's derivation
    david = DavidR(g=g, p=p, r=r)
    eq_d = david.simplify(group_context=group_context, verbose=verbose)

    domain_zz = ZZ[[david.lef.L_VAR] + david.cur.curve_hodge.lambda_symbols[1:]]
    domain_qq = QQ[[david.lef.L_VAR] + david.cur.curve_hodge.lambda_symbols[1:]]

    eq_d_zz = domain_zz.convert_from(eq_d, domain_qq)

    if time is True:
        print("Polynomials computed")
        end = perf_counter()
        print(f"------------ Time: {end - start} ------------")

    # Compare the two polynomials
    if eq_d_zz - eq_m_zz == 0:
        save_pol(eq_d_zz, filename)

        if verbose > 0:
            print("Saved polynomial. Success :)")
        if verbose > 1:
            print(f"Polynomial: {eq_d}")

        return True
    return False
