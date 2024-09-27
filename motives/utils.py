import sympy as sp
from sympy.polys.rings import PolyElement
from typing import Generator
from bisect import bisect_left
from tqdm import tqdm
import pickle
from functools import lru_cache
import os
from typing import Optional, Iterable
from functools import reduce


class Partitions:
    """
    A helper class for constructing the Mozgovoy formula.

    Parameters
    ----------
    partition : tuple[int]
        The partition to use. It should be ordered in decreasing order.

    Methods
    -------
    elements()
        Generate the elements of d(partition) for a given partition.
    a(i, j)
        The a operator used to construct the Mozgovoy formula.
    l(i, j)
        The l operator used to construct the Mozgovoy formula.
    """

    def __init__(self, partition: tuple[int]):
        self._partition = partition

    @property
    def elements(self) -> Generator[tuple[int], None, None]:
        """
        Generate the elements of d(partition) for a given partition.

        Yields
        ------
        tuple
            The elements of d(partition).
        """
        for i, el in enumerate(self._partition, 1):
            for j in range(1, el + 1):
                yield (i, j)

    def a(self, i: int, j: int) -> int:
        """
        a operator used to construct the Mozgovoy formula.

        Parameters
        ----------
        i : int
            The first index of the operator.
        j : int
            The second index of the operator.

        Returns
        -------
        int
            The output of the operator.
        """
        return self._partition[i - 1] - j

    def l(self, i: int, j: int) -> int:
        """
        l operator used to construct the Mozgovoy formula.

        Parameters
        ----------
        i : int
            The first index of the operator.
        j : int
            The second index of the operator.

        Returns
        -------
        int
            The output of the operator.
        """
        index = len(self._partition) - bisect_left(self._partition[::-1], j)
        return index - i


def generate_partitions(r: int) -> Generator[tuple[int], None, None]:
    """
    Generate all partitions of r for k = 1, 2, ..., r.

    Parameters
    ----------
    r : int
        The number to partition.

    Yields
    ------
    tuple
        A partition of r.
    """
    if r < 1:
        yield ()
        return
    for k in range(1, r + 1):
        for p in ordered_partitions(r, k):
            yield p


def multinomial_coeff(lst: list[int]) -> int:
    """
    Calculate the multinomial coefficient of a list.
    multinomial_coeff([a, b, c]) is equivalent to (a + b + c)! / (a! * b! * c!).

    Parameters
    ----------
    lst : list[int]
        The list of numbers to calculate the multinomial coefficient.

    Returns
    -------
    int
        The multinomial coefficient of the list.
    """
    res, i = 1, sum(lst)
    i0 = lst.index(max(lst))
    for a in lst[:i0] + lst[i0 + 1 :]:
        for j in range(1, a + 1):
            res *= i
            res //= j
            i -= 1
    return res


def partitions(n: int, k: int) -> Generator[tuple[int], None, None]:
    """
    Generate all partitions of n into k parts.

    Parameters
    ----------
    n : int
        The number to partition.
    k : int
        The number of parts.

    Yields
    ------
    tuple
        A partition of n into k parts.
    """
    if k < 1:
        return
    if k == 1:
        yield (n,)
        return
    for i in range(n + 1):
        for result in partitions(n - i, k - 1):
            yield (i,) + result


@lru_cache(maxsize=None)
def ordered_partitions(n: int, k: int, prev: int = 1) -> tuple[tuple[int]]:
    """
    Generate all ordered partitions of n into k parts, and return them as a tuple of tuples.

    Parameters
    ----------
    n : int
        The number to partition.
    k : int
        The number of parts.

    Returns
    -------
    tuple of tuples
        A tuple of ordered partitions of n into k parts.
    """
    if k < 1:
        return ()
    if k == 1:
        return ((n,),)

    partitions = []
    for i in range(prev, n // k + 1):
        for result in ordered_partitions(n - i, k - 1, i):
            partitions.append(result + (i,))

    return tuple(partitions)


def lambda_of_adams_expansion(lambda_vars: list[sp.Expr], n: int, k: int) -> sp.Expr:
    """
    Returns the polynomial of lambda operators equivalent to λ_n(ψ_k(exp))

    Parameters
    ----------
    lambda_vars : list[sp.Expr]
        The list of lambda variables to use.
    n : int
        The degree of the lambda operator.
    k : int
        The degree of the adams operator.

    Returns
    -------
    The polynomial of lambda operators equivalent to λ_n(ψ_k(exp)).
    """
    x = sp.Symbol("x")
    if k == 0:
        return sp.Integer(0)
    if k == 1:
        return lambda_vars[n]
    if k == 2:
        root = -1
    elif k > 2:
        root = sp.rootof(x**k - 1, k - 1)

    exp = (
        (
            sum(
                sp.prod(lambda_vars[j] for j in p)
                * root ** sum(j * c for j, c in enumerate(p))
                for p in partitions(k * n, k)
            )
        )
        .expand()
        .simplify()
    )
    return exp


def subs_variable(
    expr: sp.Expr,
    var: sp.Symbol,
    subs: int | sp.Symbol,
    denom_var: sp.Symbol,
    domain: sp.Domain,
    *,
    verbose: int = 0,
) -> PolyElement:
    """
    Function that cancels the necessary terms to avoid indeterminate forms
    before substituting a variable with a value.

    Parameters
    ----------
    expr : sp.Expr
        The expression to cancel.
    var : sp.Symbol
        The variable that is going to be substituted.
    subs : int | sp.Symbol
        The value to substitute the variable with.
    domain : sp.Domain
        The domain of the polynomial to return without denom_var.
    denom_var : sp.Symbol
        The variable that is going to be canceled in the last step.
    verbose : int
        How much to print when deriving the formula. 0 is nothing at all,
        1 is only sentences to see progress, 2 is also intermediate formulas.

    Returns
    -------
    sp.Expr
        The expression with the necessary terms canceled and the variable substituted.
    """

    collected: dict[sp.Expr, sp.Expr] = sp.collect(expr, (var - subs), evaluate=False)

    # Create a list of coefficients.
    max_power = max(-power.as_base_exp()[1] for power in collected.keys())
    coeffs: list[sp.Expr] = [0] * (max_power + 1)

    for power, coeff in collected.items():
        if power == 1:
            coeffs[0] += coeff.xreplace({var: subs})
        else:
            exp = power.as_base_exp()[1]
            if exp > 0:
                print(f"There is a term with power {exp} in the expression.")
                # We ignore it as it would be 0 when substituted
                continue
            coeffs[-exp] = coeff

    # Iterate the dictionary from the highest power to the lowest. Cancel as
    # many terms as possible and add to the respective power.
    iterable = tqdm(range(max_power, 0, -1)) if verbose > 0 else range(max_power, 0, -1)
    for power in iterable:
        coeff = coeffs[power]

        if coeff == 0:
            continue

        # Get a common denominator and save it for later.
        com_coeff = sp.together(coeff)
        numerator, denominator = sp.fraction(com_coeff)

        # Simplify the numerator by dividing by var - subs
        quotient: sp.Poly = sp.poly(numerator, var)
        divisor: sp.Poly = sp.poly(var - subs, var, domain=quotient.domain)
        n = power
        while quotient.eval(subs) == 0:
            quotient = quotient.exquo(divisor)
            n -= 1

        if n == power:
            raise Exception(f"The coefficient of degree {n} couldn't be divided")

        # Convert the quotient back to an expression
        # Parallelizable
        quot_as_dict = quotient.rep.to_dict()
        symbols = next(iter(quot_as_dict.values())).ring.symbols
        quotient = sp.Add(
            *[
                var**exp * expr_from_pol(coeff, symbols)
                for (exp,), coeff in quot_as_dict.items()
            ]
        )

        # Add the term to the respective power
        if n >= 0:
            coeffs[n] += quotient / denominator
        else:
            raise Exception("The power of the term in which to add is negative")

    result = coeffs[0].xreplace({var: subs})

    # Cancel the denominator of the expression
    cancelled_result = cancel(result, domain, denom_var)

    return cancelled_result


def expr_from_pol(
    pol: PolyElement, symbols: Optional[Iterable[sp.Symbol]] = None
) -> sp.Expr:
    """
    Convert a polynomial into an expression.

    Parameters
    ----------
    pol : PolyElement
        The polynomial to convert.
    symbols : Optional[Iterable[sp.Symbol]]
        The symbols to use in the expression. If None, it will use the symbols
        of the polynomial.

    Returns
    -------
    sp.Expr
        The expression equivalent to the polynomial.
    """
    dict_rep = pol.as_expr_dict()
    if symbols is None:
        symbols = pol.ring.symbols
    result = []

    for monom, coeff in dict_rep.items():
        term = [coeff]
        for g, m in zip(symbols, monom):
            if m:
                term.append(sp.Pow(g, m))

        result.append(sp.Mul(*term, evaluate=False))

    return sp.Add(*result, evaluate=False)


def unify_terms(
    expr: sp.Expr, domain: sp.Domain, var_den: sp.Symbol
) -> tuple[sp.Poly, sp.Poly]:
    """
    Simplify an expression by combining the terms with a common denominator.

    Parameters
    ----------
    expr : sp.Expr
        The expression to simplify.
    domain : sp.Domain
        The domain of the polynomial.
    var_den : sp.Symbol
        The variable that is in the denominator.

    Returns
    -------
    tuple[sp.Poly, sp.Poly]
        The numerator and the denominator of the expression.
    """

    one_num = sp.poly(1, var_den, domain=domain)
    one_den = sp.poly(1, var_den)
    n_symbols = len(domain.ring.symbols)
    positions = {symbol: i for i, symbol in enumerate(domain.ring.symbols)}

    def _together(expr: sp.Expr) -> tuple[PolyElement, PolyElement]:
        """
        Helper function to simplify an expression by combining the terms with a common denominator.
        """
        if expr.is_Atom:
            if expr.is_Integer:
                return (
                    sp.Poly.from_dict(
                        {(0,): domain.ring.from_dict({(0,) * n_symbols: expr})},
                        var_den,
                        domain=domain,
                    ),
                    one_den,
                )
            if isinstance(expr, sp.Rational):
                return (
                    sp.Poly.from_dict(
                        {(0,): domain.ring.from_dict({(0,) * n_symbols: expr.p})},
                        var_den,
                        domain=domain,
                    ),
                    sp.Poly.from_dict({(0,): expr.q}, var_den),
                )
            # It is a symbol
            if expr == var_den:
                return (
                    sp.Poly.from_dict({(1,): domain.ring.one}, var_den, domain=domain),
                    one_den,
                )
            # It is a symbol different from var_den
            return (
                sp.Poly.from_dict({(0,): domain(expr)}, var_den, domain=domain),
                one_den,
            )

        elif expr.is_Add:
            # Parallelizable
            # Generator of tuples of the form (numerator, denominator)
            terms = tuple(map(_together, sp.Add.make_args(expr)))

            common_denom = one_den
            for _, denom in terms:
                common_denom = sp.lcm(common_denom, denom)

            if common_denom.is_one:
                return sum(numer for numer, _ in terms), one_den
            if common_denom.is_integer:
                common_denom = one_den * common_denom

            # Unify the terms
            new_numer = sum(
                numer * denom_to_numer(common_denom.exquo(denom), var_den, domain)
                for numer, denom in terms
            )

            com_factor = new_numer.gcd(denom_to_numer(common_denom, var_den, domain))
            return new_numer.exquo(com_factor), common_denom.exquo(
                numer_to_denom(com_factor, var_den, domain)
            )

        elif expr.is_Pow:
            base_num, base_den = _together(expr.base)

            exp = expr.exp

            if exp > 0:
                return base_num.pow(exp), base_den.pow(exp)
            return (
                denom_to_numer(base_den, var_den, domain).pow(-exp),
                numer_to_denom(base_num, var_den, domain).pow(-exp),
            )

        elif expr.is_Mul:
            if all(
                term.is_Atom or (term.is_Pow and term.base.is_Atom)
                for term in expr.args
            ):
                # If it is a monomial
                numer, denom = to_poly_monomial(expr, var_den, domain, positions)
                return numer, denom

            # Generator of tuples of the form (numerator, denominator)
            terms = map(_together, expr.args)
            numer, denom = zip(*(term for term in terms))

            new_numer, new_denom = reduce(
                lambda x, y: x.mul(y), numer, one_num
            ), reduce(lambda x, y: x.mul(y), denom, one_den)

            return new_numer, new_denom

        else:
            raise NotImplementedError(f"together not implemented for {expr}")

    return _together(expr)


def to_poly_monomial(
    monomial: sp.Expr,
    var: sp.Symbol,
    domain: sp.Domain,
    positions: dict[sp.Symbol, int],
) -> sp.Poly:
    """
    Convert a monomial expression to a polynomial.

    Parameters
    ----------
    monomial : sp.Expr
        The monomial to convert.
    var : sp.Symbol
        The variable of the monomial.
    domain : sp.Domain
        The domain of the polynomial.
    positions : dict[sp.Symbol, int]
        The positions of the variables in the polynomial ring.

    Returns
    -------
    sp.Poly
        The sympy polynomial equivalent to the monomial.
    """
    ring = domain.ring
    n_symbols = len(ring.symbols)
    exps_dom = [0] * n_symbols
    exp_var = 0
    coeff_num = 1
    coeff_den = 1

    for factor in monomial.as_ordered_factors():
        if factor.is_Integer:
            coeff_num *= factor
        elif factor.is_Rational:
            coeff_num *= factor.p
            coeff_den *= factor.q
        elif factor.is_Pow:
            base, exp = factor.as_base_exp()
            if base == var:
                exp_var = exp
            else:
                if base.is_Integer and exp < 0:
                    coeff_den *= base**-exp
                exps_dom[positions[base]] = sp.ZZ(exp)
        else:
            if factor == var:
                exp_var = 1
            else:
                exps_dom[positions[factor]] = sp.ZZ(1)

    if exp_var < 0:
        return sp.Poly.from_dict(
            {(0,): ring.from_dict({tuple(exps_dom): sp.ZZ(coeff_num)})},
            var,
            domain=domain,
        ), sp.Poly.from_dict({(-exp_var,): coeff_den}, var)
    else:
        return sp.Poly.from_dict(
            {(exp_var,): ring.from_dict({tuple(exps_dom): sp.ZZ(coeff_num)})},
            var,
            domain=domain,
        ), sp.Poly.from_dict({(0,): coeff_den}, var)


def denom_to_numer(poly: sp.Poly, var: sp.Symbol, domain: sp.Domain) -> sp.Poly:
    """
    Helper function for `to_poly_monomial` that convert a polynomial element in the domain
    of the denominator to the domain of the numerator.

    Parameters
    ----------
    poly : PolyElement
        The polynomial element to convert.
    var : sp.Symbol
        The variable of the denominator.
    domain : sp.Domain
        The domain of the numerator.

    Returns
    -------
    sp.Poly
        The polynomial element converted to the domain of the numerator.
    """
    ring = domain.ring
    n_symbols = len(ring.symbols)

    coeff_dict = {
        exp: ring.from_dict({(0,) * n_symbols: coeff})
        for exp, coeff in poly.rep.to_dict().items()
    }
    return sp.Poly.from_dict(coeff_dict, var, domain=domain)


def numer_to_denom(poly: sp.Poly, var: sp.Symbol, domain: sp.Domain) -> sp.Poly:
    """
    Helper function for `to_poly_monomial` that convert a polynomial element in the domain
    of the numerator to the domain of the denominator.

    Parameters
    ----------
    poly : PolyElement
        The polynomial element to convert.
    var : sp.Symbol
        The variable of the denominator.
    domain : sp.Domain
        The domain of the numerator.

    Returns
    -------
    sp.Poly
        The polynomial element converted to the domain of the denominator.
    """
    length = len(domain.ring.symbols)

    terms = poly.rep.to_dict().items()
    coeff_dict = {exp: coeff.as_expr_dict()[(0,) * length] for exp, coeff in terms}

    return sp.Poly.from_dict(coeff_dict, var)


def cancel(expr: sp.Expr, domain: sp.Domain, var: sp.Symbol) -> PolyElement:
    """
    Cancel an expression's denominator.

    Parameters
    ----------
    expr : sp.Expr
        The expression to cancel.
    domain : sp.Domain
        The domain of the polynomial.
    var : sp.Symbol
        The variable that is going to be canceled.

    Returns
    -------
    PolyElement
        The polynomial with the denominator canceled.
    """
    numer, denom = unify_terms(expr, domain, var)
    if denom == sp.poly(-1, var):
        numer = -numer

    new_domain = domain.ring.domain[(var,) + domain.ring.symbols]
    var_poly = new_domain(var)

    simplify1 = new_domain.ring.add(
        new_domain.convert_from(coeff, domain) * var_poly**exp
        for (exp,), coeff in numer.rep.to_dict().items()
    )

    return simplify1


def expand_variable_1mt(var: sp.Symbol):
    """
    Helper function used in a sp.replace that returns a function
    that sets up the expression to apply a collect for the expression
    (1 - var). It is different from the expand_variable function because
    we don't want to expand the expression where (1 - t) is present.

    Parameters
    ----------
    var : sp.Symbol
        The variable that should be expanded.

    Returns
    -------
    (sp.Expr) -> bool
        The function that selects the terms to replace in the sp.replace.
    (sp.Expr) -> sp.Expr
        The function that makes the substitution in the sp.replace.
    """

    def matcher_func(expr: sp.Expr) -> bool:
        if expr.is_Pow:
            base, exp = expr.as_base_exp()
            return (base == (1 - var) or base == (var - 1)) and exp < 0
        return False

    def subs_func(expr: sp.Expr) -> sp.Expr:
        to_dist = []
        not_to_dist = []

        for term in expr.args:
            if term.is_Pow:
                # If it is a power, we will consider just the base and then
                # raise it to the respective exponent
                base, exp = term.as_base_exp()
            else:
                base = term
                exp = 1

            if base.is_Add:
                # This means it is a sum inside the product
                for subterm in base.args:
                    if subterm.find(matcher_func):
                        # This means that we should apply distributive property
                        # because it is of the form x * (y + z / (1 - var) ** k)
                        if exp > 1:
                            term = sp.expand_multinomial(term, deep=False)
                            term = sp.expand_power_base(term, force=True)

                        to_dist.append(term)
                        break
                else:
                    # This sumation does not contain the term (1 - var) ** k
                    # in any of its terms, so we don't need to apply the
                    # distributive property
                    not_to_dist.append(term)
            else:
                not_to_dist.append(term)

        if not to_dist:
            return expr

        to_dist = sp.expand_mul(sp.Mul(*to_dist), deep=False)
        not_to_dist = sp.Mul(*not_to_dist)
        return sp.Add(*[not_to_dist * term for term in to_dist.args])

    def getter_func(expr: sp.Expr) -> bool:
        return expr.is_Mul

    return getter_func, subs_func


def expand_variable_1mtk(var: sp.Symbol):
    """
    Helper function used in a sp.replace that returns a function that expands
    1 / (1 - var ** k) into 1 / ((1 - var) * (1 + var + ... + var ** (k - 1))).

    Parameters
    ----------
    var : sp.Symbol
        The variable that should be expanded.

    Returns
    -------
    (sp.Expr) -> bool
        The function that selects the terms to replace in the sp.replace.
    (sp.Expr) -> sp.Expr
        The function that makes the substitution in the sp.replace.
    """

    # Function to select the terms to replace
    def getter_func(expr: sp.Expr) -> bool:
        if expr.is_Pow:
            base, exp = expr.as_base_exp()
            if exp < 0 and base.is_Add:
                terms = base.as_ordered_terms()
                if len(terms) == 2 and base.xreplace({var: 1}) == 0:
                    if terms[0] == 1:
                        # 1 - var ** k
                        exp2 = terms[1].args[1].as_base_exp()[1]
                    elif terms[1] == 1:
                        # - var ** k + 1
                        exp2 = terms[0].args[1].as_base_exp()[1]
                    elif terms[0] == -1:
                        # -1 + var ** k
                        exp2 = terms[1].as_base_exp()[1]
                    elif terms[1] == -1:
                        # var ** k - 1
                        exp2 = terms[0].as_base_exp()[1]
                    else:
                        return False
                    return exp2 > 1
        return False

    # Function to replace matching subexpressions
    def subs_func(expr: sp.Expr) -> sp.Expr:
        base, exp = expr.as_base_exp()
        terms = base.as_ordered_terms()
        if terms[0] == 1:
            # 1 - var ** k
            exp2 = terms[1].args[1].as_base_exp()[1]
        elif terms[1] == 1:
            # - var ** k + 1
            exp2 = terms[0].args[1].as_base_exp()[1]
        elif terms[0] == -1:
            # -1 + var ** k
            exp2 = terms[1].as_base_exp()[1]
        elif terms[1] == -1:
            # var ** k - 1
            exp2 = terms[0].as_base_exp()[1]

        return ((1 - var) * sp.Add(*[var**i for i in range(exp2)])) ** exp

    return getter_func, subs_func


def expand_variable(var: sp.Symbol):
    """
    Helper function used in a sp.replace that returns a function
    that expands the expression if it contains the variable var.

    Parameters
    ----------
    var : sp.Symbol
        The variable that should be expanded.

    Returns
    -------
    (sp.Expr) -> bool
        The function that selects the terms to replace in the sp.replace.
    (sp.Expr) -> sp.Expr
        The function that makes the substitution in the sp.replace.
    """

    def subs_func(expr: sp.Expr) -> sp.Expr:
        if expr.is_Mul:
            if expr.has(var):
                # Split the expression into terms with and without the variable,
                # expand the terms with the variable, and recombine them.
                terms_w_var = sp.Mul(*[x for x in expr.args if x.has(var)])
                terms_wo_var = sp.Mul(*[x for x in expr.args if not x.has(var)])
                expanded_vars = sp.expand(terms_w_var, force=True)
                if expanded_vars.is_Add:
                    return sum(
                        term_var * terms_wo_var for term_var in expanded_vars.args
                    )
                else:
                    return expanded_vars * terms_wo_var

        if expr.is_Pow:
            exp = expr.exp
            if exp > 1 and expr.has(var):
                new_expr = sp.expand_multinomial(expr, deep=False)
                new_expr = sp.expand_power_base(new_expr, force=True)
                return new_expr

        return expr

    def getter_func(expr: sp.Expr) -> bool:
        return expr.is_Mul or expr.is_Pow

    return getter_func, subs_func


def save_pol(pol: PolyElement, filename: str) -> None:
    """
    Save a polynomial to a pickle file. It will save the polynomial as a
    dictionary with the domain, the variables, and the coefficients.

    Parameters
    ----------
    pol : PolyElement
        The polynomial to save.
    filename : str
        The filename to save the polynomial to.

    Returns
    -------
    None
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    data = pol.as_expr_dict()
    data["domain"] = pol.ring.domain
    data["vars"] = pol.ring.symbols

    with open(filename, "wb") as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_pol(filename: str) -> PolyElement:
    """
    Load a polynomial from a pickle file. If the file has the domain and the
    variables, it will use them to load the polynomial. Otherwise, it will
    assume the domain is ZZ and the variables are [L, a1(h_C), a2(h_C), ...].

    Parameters
    ----------
    filename : str
        The filename to load the polynomial from.

    Returns
    -------
    PolyElement
        The polynomial loaded from the file.
    """
    with open(filename, "rb") as fp:
        data: dict = pickle.load(fp)

    domain: sp.Domain = data.pop("domain", None)
    vars: list[sp.Symbol] = data.pop("vars", None)

    # If the domain or the variables are not present, we will assume that the
    # polynomial is over the integers and that the variables are the following
    if domain is None or vars is None:
        length = len(next(iter(data.keys())))
        vars = [sp.Symbol("L")] + [sp.Symbol(f"a{i}(h_C)") for i in range(1, length)]
        r, *_ = sp.ring(vars, domain=sp.ZZ)
    else:
        r, *_ = sp.ring(vars, domain=domain)

    return r.from_dict(data)
