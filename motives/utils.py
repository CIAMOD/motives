import sympy as sp
from sympy.polys.rings import PolyElement
from typing import Callable, Generator
from bisect import bisect_left
from tqdm import tqdm
import pickle
from functools import lru_cache, reduce
from operator import mul
import os
from typing import Optional, Iterable
from functools import reduce
from threading import Lock


class SingletonMeta(type):
    """
    A thread-safe Singleton metaclass.

    This metaclass ensures that a class using it will only ever have one instance
    (singleton) throughout the lifetime of the program, regardless of how many times
    the class is instantiated. If an instance of the class does not exist, it creates
    one, otherwise, it returns the existing instance.

    Attributes:
    -----------
    _instances : dict
        A dictionary that stores instances of classes using this metaclass as keys,
        ensuring only one instance per class.

    _lock : threading.Lock
        A threading lock object that ensures thread safety when creating the
        singleton instance.

    Methods:
    --------
    __call__(cls, *args, **kwargs):
        Overrides the default behavior of the `__call__` method to control how
        instances are created. It ensures only one instance of the class is created,
        even in a multithreaded environment.

    References:
    -----------
    The code of this class is based on the Singleton design pattern example provided in
    [1] https://refactoring.guru/design-patterns/singleton/python/example
    """

    _instances = {}
    _lock = Lock()

    def __call__(cls, *args, **kwargs):
        """
        Controls the instantiation of classes using this metaclass.

        This method ensures that only one instance of the class is created by
        checking the `_instances` dictionary. If the class does not already have
        an instance, it creates one. The method uses a thread lock to ensure that
        multiple threads do not attempt to create instances simultaneously, ensuring
        thread safety.

        Parameters:
        -----------
        *args : tuple
            Positional arguments passed to the class constructor.
        **kwargs : dict
            Keyword arguments passed to the class constructor.

        Returns:
        --------
        object
            The singleton instance of the class.
        """
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Partitions:
    """
    A class for handling Young diagrams associated to a
    partition of an integer r into k summands.

    A k-partition of n is represented by a nonincreasing sequence of integers
        a_1>=a_2>=...>=a_k>=0
    such that
        a_1+a_2+...+a_k=r

    Given a partition (a_1,...,a_k), A Young diagram associated to a partition
    is constructed as a finite collection of piled boxes organized in rows where
    the number of boxes for each row is given by the integers a_i. For example,
    the following partition of 8, (3,2,2,1), has as Youn diagram

        XXXX
        XX
        XX
        X

    whose elements are designated as pairs (i,j) denoting the box of the
    diagram in row i and column j.

    This class permits computing some combinatorial functions associated to the Young
    diagram of a given partition.

    Parameters
    ----------
    partition : tuple[int]
        The partition to use. It should be ordered in descending order.

    Methods
    -------
    elements()
        Generate the elements of the Young diagram of the partition.
    a(i, j)
        Arm length of an element (i,j) in the Young diagram of the partition.
        It is the number of elements to the right of (i,j) in the diagram.
    l(i, j)
        Leg length of an element (i,j) in the Young diagram of the partition.
        It is the number of elements below (i,j) in the diagram.
    h(i,j)
        hook length of an element (i,j) in the Young diagram of the partition.
        It is the number of elements to the right or below (i,j), including itself, in the diagram.
        It equals a(i,j)+l(i,j)+1.
    """

    def __init__(self, partition: tuple[int]):
        self._partition = partition

    @property
    def elements(self) -> Generator[tuple[int], None, None]:
        """
        Generate the elements of d(partition) for a given partition, i.e.,
        the points in the Young or Ferres diagram of a partition.

        For example, the partition (2,2,1) of 5 with Young diagram
            XX
            XX
            X
        has (1,1), (1,2), (2,1), (2,2) and (3,1) as elements.

        Yields
        ------
        tuple
            The elements of d(partition). Each element is a pair (i,j), where
            i indicates the row and j the column (starting from 1).
        """
        for i, el in enumerate(self._partition, 1):
            for j in range(1, el + 1):
                yield (i, j)

    def a(self, i: int, j: int) -> int:
        """
        Arm length of an element (i,j) in the Young diagram of a partition.
        Denoted a(i,j), it is the number of elements to the right of (i,j) in the diagram.

        Parameters
        ----------
        i : int
            The first index of the operator.
        j : int
            The second index of the operator.

        Returns
        -------
        int
            The output of the operator a(i,j).
        """
        return self._partition[i - 1] - j

    def l(self, i: int, j: int) -> int:
        """
        Leg length of an element (i,j) in the Young diagram of the partition.
        Denoted l(i,j), it is the number of elements below (i,j) in the diagram.

        Parameters
        ----------
        i : int
            The first index of the operator.
        j : int
            The second index of the operator.

        Returns
        -------
        int
            The output of the operator l(i,j).
        """
        index = len(self._partition) - bisect_left(self._partition[::-1], j)
        return index - i

    def h(self, i: int, j: int) -> int:
        """
        Hook length of an element (i,j) in the Young diagram of the partition.
        Denoted h(i,j), it is the number of elements to the right or below (i,j),
        including itself, in the diagram. It equals
            a(i,j)+l(i,j)+1.

        Parameters
        ----------
        i : int
            The first index of the operator.
        j : int
            The second index of the operator.

        Returns
        -------
        int
            The output of the operator h(i,j).
        """
        return self.a(i, j) + self.l(i, j) + 1


def all_partitions(r: int) -> Generator[tuple[int], None, None]:
    """
    Generate all partitions of r in k summands for k = 1, 2, ..., r.

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


def preorder_traversal(expr):
    yield expr
    if isinstance(expr, sp.Basic):
        for arg in expr.args:
            yield from preorder_traversal(arg)


def prod(iterable: Iterable) -> object:
    """
    Calculate the product of an iterable.

    Parameters
    ----------
    iterable : Iterable[Any]
        The iterable to calculate the product of.

    Returns
    -------
    Any
        The product of the iterable
    """
    return reduce(mul, iterable, 1)


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


@lru_cache(maxsize=None)
def partitions(n: int, k: int, minimum: int = 0) -> tuple[tuple[int]]:
    """
    Generate all partitions of n into k parts with each part being at least `minimum`.

    Parameters
    ----------
    n : int
        The number to partition.
    k : int
        The number of parts.
    minimum : int, optional
        The minimum value for each part in the partition (default is 0).

    Returns
    -------
    tuple
        A partition of n into k parts with each part being at least `minimum`.
    """
    if k < 1:
        return ()
    if k == 1:
        if n >= minimum:
            return ((n,),)
        return ()

    parts = []
    for i in range(minimum, n + 1):
        for result in partitions(n - i, k - 1, minimum):
            parts.append((i,) + result)

    return tuple(parts)


def sp_decimal_part(p: int, q: int) -> sp.Rational:
    """
    Calculate the decimal part of a fraction.

    Parameters
    ----------
    p : int
        The numerator of the fraction.
    q : int
        The denominator of the fraction.

    Returns
    -------
    sp.Rational
        The decimal part of the fraction.
    """
    return sp.Rational(p, q) - sp.Rational(p // q)


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


def cancel(expr: sp.Expr, domain: sp.Domain, var: sp.Symbol) -> PolyElement:
    """
    Cancel the denominator of a given rational expression over a specified domain.

    This function cancels out the denominator of a rational expression, ensuring that
    the returned result is in the form of a polynomial with the denominator removed.
    The cancellation is performed over the specified polynomial domain, and the
    result is returned as a `PolyElement` object.

    Args:
    -----
    expr : sp.Expr
        The symbolic expression from SymPy to be simplified by canceling its denominator.
    domain : sp.Domain
        The domain over which the polynomial is defined. This is needed to handle
        the algebraic properties of the expression during the cancellation process.
    var : sp.Symbol
        The variable with respect to which the cancellation will be applied.

    Returns:
    --------
    PolyElement
        A polynomial in `PolyElement` form with the denominator canceled. This is
        equivalent to the numerator of the original expression after cancellation.

    Raises:
    -------
    ValueError
        If the expression is not a valid rational expression or cannot be processed.
    """
    numer, denom = cancel_poly(expr, domain, var)
    if denom == sp.poly(-1, var):
        numer = -numer

    # Convert from sympy poly to PolyElement
    new_domain = domain.ring.domain[(var,) + domain.ring.symbols]
    var_poly = new_domain(var)

    cancelled_expr = new_domain.ring.add(
        new_domain.convert_from(coeff, domain) * var_poly**exp
        for (exp,), coeff in numer.rep.to_dict().items()
    )

    return cancelled_expr


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

    if verbose > 0:
        print("Cancelling the denominator of the expression.")
        if verbose > 1:
            print(f"Expression to cancel: {result}")

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
    symbols = symbols or pol.ring.symbols
    result = []

    for monom, coeff in dict_rep.items():
        term = [coeff]
        for g, m in zip(symbols, monom):
            if m:
                term.append(sp.Pow(g, m))

        result.append(sp.Mul(*term, evaluate=False))

    return sp.Add(*result, evaluate=False)


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

    # Get the exponents of the symbols, the exponent of the variable, and the coefficient
    for factor in monomial.as_ordered_factors():
        if factor.is_Integer:
            # Contributes to the numerator coefficient
            coeff_num *= factor
        elif factor.is_Rational:
            # Separate into numerator and denominator and
            # multiply them into the respective coefficients
            coeff_num *= factor.p
            coeff_den *= factor.q
        elif factor.is_Pow:
            base, exp = factor.as_base_exp()
            if base == var:
                # var ** exp
                exp_var = exp
            else:
                if base.is_Integer and exp < 0:
                    # integer ** -integer
                    coeff_den *= base**-exp
                else:
                    # symbol ** exp
                    exps_dom[positions[base]] = sp.ZZ(exp)
        else:
            if factor == var:
                exp_var = 1
            else:
                exps_dom[positions[factor]] = sp.ZZ(1)

    # Create the polynomial: coeff_num * var ** exp_var * prod(symbols ** exps_dom)
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


def cancel_poly(
    expr: sp.Expr, domain: sp.Domain, var_den: sp.Symbol
) -> tuple[sp.Poly, sp.Poly]:
    """
    Cancel an expression's denominator.

    It must be an expression which only has one variable in the denominator, and its
    numerator has to be able to be converted to a polynomial. It performs the cancellation
    by recursively combining the terms with a common denominator, converting to a
    polynomial both the numerator and the denominator and dividing them.

    Parameters
    ----------
    expr : sp.Expr
        The expression to cancel.
    domain : sp.Domain
        The domain of the polynomial.
    var_den : sp.Symbol
        The variable that is going to be canceled.

    Returns
    -------
    PolyElement
        The polynomial with the denominator canceled.
    """
    one_num = sp.poly(1, var_den, domain=domain)
    one_den = sp.poly(1, var_den)
    n_symbols = len(domain.ring.symbols)
    positions: dict[PolyElement, int] = {
        symbol: i for i, symbol in enumerate(domain.ring.symbols)
    }

    def denom_to_numer(poly: sp.Poly) -> sp.Poly:
        """
        Helper function for `to_poly_monomial` that convert a polynomial element in the domain
        of the denominator to the domain of the numerator.

        Parameters
        ----------
        poly : sp.Poly
            The polynomial element to convert.

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
        return sp.Poly.from_dict(coeff_dict, var_den, domain=domain)

    def numer_to_denom(poly: sp.Poly) -> sp.Poly:
        """
        Helper function for `to_poly_monomial` that convert a polynomial element in the domain
        of the numerator to the domain of the denominator.

        Parameters
        ----------
        poly : sp.Poly
            The polynomial element to convert.

        Returns
        -------
        sp.Poly
            The polynomial element converted to the domain of the denominator.
        """
        length = len(domain.ring.symbols)

        terms = poly.rep.to_dict().items()
        coeff_dict = {exp: coeff.as_expr_dict()[(0,) * length] for exp, coeff in terms}

        return sp.Poly.from_dict(coeff_dict, var_den)

    def _together(expr: sp.Expr) -> tuple[PolyElement, PolyElement]:
        """Helper function that is called recursively It is the one that does the work.

        Args:
            expr: The expression to simplify.

        Returns:
            The numerator and the denominator of the expression, simplified.
        """
        if expr.is_Atom:
            # Just convert it to the correct polynomial domain and return it
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

            # Find the common denominator
            common_denom = one_den
            for _, denom in terms:
                common_denom = sp.lcm(common_denom, denom)

            if common_denom.is_one:
                return sum(numer for numer, _ in terms), one_den
            if common_denom.is_integer:
                common_denom = one_den * common_denom

            # Unify the terms
            new_numer = sum(
                numer * denom_to_numer(common_denom.exquo(denom))
                for numer, denom in terms
            )

            # Find the common factor and divide by it to simplify the expression
            com_factor = new_numer.gcd(denom_to_numer(common_denom))
            return new_numer.exquo(com_factor), common_denom.exquo(
                numer_to_denom(com_factor)
            )

        elif expr.is_Pow:
            base_num, base_den = _together(expr.base)

            exp = expr.exp

            if exp > 0:
                return base_num.pow(exp), base_den.pow(exp)
            return (
                denom_to_numer(base_den).pow(-exp),
                numer_to_denom(base_num).pow(-exp),
            )

        elif expr.is_Mul:
            if all(
                term.is_Atom or (term.is_Pow and term.base.is_Atom)
                for term in expr.args
            ):
                # If it is a monomial, we convert it directly for efficiency
                numer, denom = to_poly_monomial(expr, var_den, domain, positions)
                return numer, denom

            # Generator of tuples of the form (numerator, denominator)
            terms = map(_together, expr.args)
            numer, denom = zip(*(term for term in terms))

            # Multiply the individual numerators and denominators to get the new ones
            new_numer, new_denom = reduce(
                lambda x, y: x.mul(y), numer, one_num
            ), reduce(lambda x, y: x.mul(y), denom, one_den)

            return new_numer, new_denom

        else:
            raise NotImplementedError(f"together not implemented for {expr}")

    return _together(expr)


def expand_variable_1mt(
    var: sp.Symbol,
) -> tuple[Callable[[sp.Expr], bool], Callable[[sp.Expr], sp.Expr]]:
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
        """
        Checks if an expression matches a specific form: 1 / (1 - var) ** k or 1 / (var - 1) ** k,
        where `k` is a negative exponent. This is used to identify expressions that need to be
        expanded.

        Args:
        -----
        expr : sp.Expr
            The SymPy expression to check.

        Returns:
        --------
        bool
            True if the expression matches the desired form and should be expanded,
            False otherwise.
        """
        if expr.is_Pow:
            base, exp = expr.as_base_exp()
            return (base == (1 - var) or base == (var - 1)) and exp < 0
        return False

    def subs_func(expr: sp.Expr) -> sp.Expr:
        """
        Expands an expression selectively by applying the distributive property
        to certain terms that match a pattern identified by `matcher_func`.

        Args:
        -----
        expr : sp.Expr
            The SymPy expression to expand.

        Returns:
        --------
        sp.Expr
            The expanded expression where distributive properties have been applied
            to matching terms.
        """
        to_dist = []  # Terms that need distributive expansion
        not_to_dist = []  # Terms that do not require expansion

        for term in expr.args:
            if term.is_Pow:
                # Separate base and exponent if the term is a power
                base, exp = term.as_base_exp()
            else:
                base = term
                exp = 1

            if base.is_Add:
                # If it's an addition (e.g., a sum inside the product)
                for subterm in base.args:
                    if subterm.find(matcher_func):
                        # Apply the distributive property if matcher_func identifies a match
                        if exp > 1:
                            term = sp.expand_multinomial(term, deep=False)
                            term = sp.expand_power_base(term, force=True)

                        to_dist.append(term)
                        break
                else:
                    # If no match is found, add the term to the non-distributive list
                    not_to_dist.append(term)
            else:
                not_to_dist.append(term)

        if not to_dist:
            return expr  # Return original expression if no distributive expansion is needed

        # Expand terms that require distributive property and multiply them by non-distributive terms
        to_dist = sp.expand_mul(sp.Mul(*to_dist), deep=False)
        not_to_dist = sp.Mul(*not_to_dist)
        return sp.Add(*[not_to_dist * term for term in to_dist.args])

    def getter_func(expr: sp.Expr) -> bool:
        """
        Determines if an expression is a multiplication (`Mul` in SymPy).

        Args:
        -----
        expr : sp.Expr
            The SymPy expression to check.

        Returns:
        --------
        bool
            True if the expression is a multiplication, False otherwise.

        """
        return expr.is_Mul

    return getter_func, subs_func


def expand_variable_1mtk(
    var: sp.Symbol,
) -> tuple[Callable[[sp.Expr], bool], Callable[[sp.Expr], sp.Expr]]:
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

    def getter_func(expr: sp.Expr) -> bool:
        """
        Selects expressions of the form 1 / (1 - var ** k) or equivalent,
        where `k` is a positive exponent, to determine if they should be expanded.

        This function checks if an expression matches the desired pattern and
        returns `True` if it should be expanded. The target form is
        `1 / (1 - var ** k)` or its equivalent variants, such as
        `1 / (var ** k - 1)`.

        Args:
        -----
        expr : sp.Expr
            The SymPy expression to check.

        Returns:
        --------
        bool
            True if the expression matches the form and should be expanded,
            False otherwise.
        """
        if expr.is_Pow:
            base, exp = expr.as_base_exp()
            if (
                exp < 0 and base.is_Add
            ):  # Check if base is a sum and exponent is negative
                terms = base.as_ordered_terms()  # Get ordered terms of the base
                if len(terms) == 2 and base.xreplace({var: 1}) == 0:
                    # Check if the expression simplifies to 0 when var is replaced with 1
                    if terms[0] == 1:
                        # Case: 1 - var ** k
                        exp2 = terms[1].args[1].as_base_exp()[1]
                    elif terms[1] == 1:
                        # Case: - var ** k + 1
                        exp2 = terms[0].args[1].as_base_exp()[1]
                    elif terms[0] == -1:
                        # Case: -1 + var ** k
                        exp2 = terms[1].as_base_exp()[1]
                    elif terms[1] == -1:
                        # Case: var ** k - 1
                        exp2 = terms[0].as_base_exp()[1]
                    else:
                        return False
                    return exp2 > 1  # Return True if exponent `k` is greater than 1
        return False

    def subs_func(expr: sp.Expr) -> sp.Expr:
        """
        Replaces a matching subexpression of the form `1 - var ** k` or equivalent
        with an expanded version using a sum of powers of `var`.

        This function processes expressions that match patterns such as
        `1 - var ** k` or equivalent forms (`var ** k - 1`, `- var ** k + 1`, `-1 + var ** k`),
        and replaces them with a new form that expands the power term into a sum of powers of `var`.

        Args:
        -----
        expr : sp.Expr
            The SymPy expression that contains the subexpression to be replaced.

        Returns:
        --------
        sp.Expr
            The modified expression with the matching subexpression replaced
            by an expanded sum of powers of `var`.
        """
        base, exp = expr.as_base_exp()  # Get the base and exponent of the expression
        terms = base.as_ordered_terms()  # Break down the base into ordered terms

        # Identify the form of the expression and extract the exponent `k`
        if terms[0] == 1:
            # Case: 1 - var ** k
            exp2 = terms[1].args[1].as_base_exp()[1]
        elif terms[1] == 1:
            # Case: - var ** k + 1
            exp2 = terms[0].args[1].as_base_exp()[1]
        elif terms[0] == -1:
            # Case: -1 + var ** k
            exp2 = terms[1].as_base_exp()[1]
        elif terms[1] == -1:
            # Case: var ** k - 1
            exp2 = terms[0].as_base_exp()[1]

        # Replace the matching expression with the expanded sum of powers of var
        return ((1 - var) * sp.Add(*[var**i for i in range(exp2)])) ** exp

    return getter_func, subs_func


def expand_variable(
    var: sp.Symbol,
) -> tuple[Callable[[sp.Expr], bool], Callable[[sp.Expr], sp.Expr]]:
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
        """
        Expands a mathematical expression by selectively expanding terms that contain a specific variable.
        This function handles both multiplication (`Mul`) and exponentiation (`Pow`) cases, applying the
        expansion to terms involving the specified variable, and recombining them with the rest of the expression.

        Args:
        -----
        expr : sp.Expr
            The SymPy expression to expand. It may contain terms that involve the target variable.

        Returns:
        --------
        sp.Expr
            The expanded expression. If the expression contains terms with the specified variable,
            those terms will be expanded and then recombined with the other terms in the original expression.
            If no expansion is necessary, the original expression is returned unchanged.
        """
        # Case 1: If the expression is a multiplication
        if expr.is_Mul:
            if expr.has(var):  # Check if the expression contains the variable
                # Split the expression into terms that have the variable and terms that do not
                terms_w_var = sp.Mul(
                    *[x for x in expr.args if x.has(var)]
                )  # Terms with the variable
                terms_wo_var = sp.Mul(
                    *[x for x in expr.args if not x.has(var)]
                )  # Terms without the variable
                # Expand the terms that have the variable
                expanded_vars = sp.expand(terms_w_var, force=True)
                if expanded_vars.is_Add:  # If expansion results in a sum of terms
                    # Recombine the expanded terms with the non-variable terms
                    return sum(
                        term_var * terms_wo_var for term_var in expanded_vars.args
                    )
                else:
                    return (
                        expanded_vars * terms_wo_var
                    )  # Return expanded result combined with non-variable terms

        # Case 2: If the expression is a power (exponentiation)
        if expr.is_Pow:
            exp = expr.exp  # Get the exponent
            if exp > 1 and expr.has(
                var
            ):  # Check if the exponent is greater than 1 and the expression has the variable
                # Expand the power expression using multinomial expansion and expanding the base
                new_expr = sp.expand_multinomial(expr, deep=False)
                new_expr = sp.expand_power_base(new_expr, force=True)
                return new_expr

        # If no expansion is necessary, return the original expression
        return expr

    def getter_func(expr: sp.Expr) -> bool:
        """
        Checks whether the given expression is a multiplication (`Mul`) or exponentiation (`Pow`).

        This function is used to determine if an expression should be further processed based on whether it is a multiplication or a power.

        Args:
        -----
        expr : sp.Expr
            The SymPy expression to check.

        Returns:
        --------
        bool
            True if the expression is either a multiplication (`Mul`) or an exponentiation (`Pow`),
            False otherwise.
        """
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


@lru_cache(maxsize=None)
def stirling1(n: int, k: int) -> int:
    """
    Computes the value of the (unsigned) Stirling number of the first kind, denoted S1(n, k).

    The Stirling numbers of the first kind count the number of permutations of `n` elements
    with exactly `k` disjoint cycles. These numbers satisfy the recurrence relation:

    S1(n, k) = (n - 1) * S1(n - 1, k) + S1(n - 1, k - 1)

    Base cases:
    - S1(0, 0) = 1
    - S1(n, 0) = 0 for n > 0
    - S1(0, k) = 0 for k > 0

    Args:
    -----
    n : int
        The total number of elements.
    k : int
        The number of disjoint cycles.

    Returns:
    --------
    int
        The Stirling number of the first kind for the given `n` and `k`.
    """
    if n == k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    return (n - 1) * stirling1(n - 1, k) + stirling1(n - 1, k - 1)


@lru_cache(maxsize=None)
def stirling2(n: int, k: int) -> int:
    """
    Computes the Stirling number of the second kind, denoted S2(n, k).

    The Stirling numbers of the second kind count the number of ways to partition
    `n` elements into `k` non-empty subsets. These numbers satisfy the recurrence relation:

    S2(n, k) = k * S2(n - 1, k) + S2(n - 1, k - 1)

    Base cases:
    - S2(n, n) = 1
    - S2(0, k) = 0 for k > 0
    - S2(n, 0) = 0 for n > 0

    Args:
    -----
    n : int
        The total number of elements.
    k : int
        The number of non-empty subsets.

    Returns:
    --------
    int
        The Stirling number of the second kind for the given `n` and `k`.
    """
    if n == k:
        return 1
    if n == 0 or k == 0:
        return 0
    return k * stirling2(n - 1, k) + stirling2(n - 1, k - 1)


def split_semisimple_group(ds: list[int], dim: int) -> sp.Expr:
    """
    Computes the motive of an algebraic semisimple group.

    This function calculates the motive of a semisimple algebraic group based on the input
    list of exponents and the dimension of the group. The motive is expressed in terms of
    the Lefschetz motive raised to powers related to the dimension and exponents.

    Args:
    -----
    ds : list[int]
        A list of integers, where each element represents one higher than the exponents of the group.
    dim : int
        The dimension of the algebraic group.

    Returns:
    --------
    sp.Expr
        The motive of the algebraic semisimple group, expressed as a SymPy expression.
    """
    from .grothendieck_motives.lefschetz import Lefschetz

    return Lefschetz() ** dim * sp.Mul(
        *[1 - Lefschetz() ** (-d) for d in ds],
    )
