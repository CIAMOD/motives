import sympy as sp
from typeguard import typechecked

from ..lambda_ring_context import LambdaRingContext
from ..operand.operand import Operand
from ..lambda_ring_expr import LambdaRingExpr

from sympy.printing.str import StrPrinter


class RingOperator(LambdaRingExpr, sp.Function):
    """
    Represents an abstract ring operator in an expression tree.

    The `RingOperator` class is an abstract base class that represents the three
    ring operators: lambda, sigma, and adams. It serves as a parent class for these
    specific ring operators in an expression. Each `RingOperator` has a degree and
    a child expression to which the operator is applied.

    Attributes:
    -----------
    args : tuple[sp.Integer, LambdaRingExpr]
        The arguments of the ring operator, where the first element is the degree
        (an integer) and the second element is the child expression.

    Properties:
    -----------
    degree : int
        The degree of the ring operator.
    child : LambdaRingExpr
        The child node (expression) to which the ring operator is applied.
    """

    args: tuple[sp.Integer, LambdaRingExpr]

    def _eval_is_real(self) -> bool:
        """
        Returns whether the ring operator is a real number.

        The ring operator is considered real if the child expression is real.

        Returns:
        --------
        bool
            Whether the ring operator is a real number.
        """
        return self.child.is_real

    @property
    def degree(self) -> int:
        """
        The degree of the ring operator.

        The degree is the first argument passed to the ring operator, which represents
        the power or level of the operation applied to the child node.

        Returns:
        --------
        int
            The degree of the ring operator.
        """
        return self.args[0].p

    @property
    def child(self) -> LambdaRingExpr:
        """
        The child node of the ring operator.

        The child is the expression to which the ring operator is applied. It is
        the second argument of the ring operator.

        Returns:
        --------
        LambdaRingExpr
            The child expression of the ring operator.
        """
        return self.args[1]

    def get_max_adams_degree(self) -> int:
        """
        Computes the maximum Adams degree of this tree.

        The Adams degree of the tree is calculated by multiplying the degree of this
        ring operator with the maximum Adams degree of its child expression. This
        method recursively computes the maximum value by traversing all branches of the
        tree.

        Returns:
        --------
        int
            The maximum Adams degree of the expression tree once converted to an Adams polynomial.
        """
        return self.degree * self.child.get_max_adams_degree()


class Sigma(RingOperator):
    """
    Represents the sigma ring operator in an expression tree.

    The `Sigma` class is a specific type of `RingOperator` that applies the sigma operation
    to an expression. It should be created using `Sigma(degree, child)`, where `degree`
    is the degree of the sigma operator, and `child` is the expression to which the sigma
    operator is applied.

    Methods:
    --------
    __repr__() -> str:
        Returns the string representation of the sigma operator.

    _sympystr(printer: StrPrinter) -> str:
        Returns the SymPy string representation of the sigma operator.

    get_max_groth_degree() -> int:
        Computes the maximum sigma or lambda degree needed to create a Grothendieck context
        for the expression tree.

    _to_adams(operands: set[Operand], max_adams_degree: int, as_symbol: bool = False) -> sp.Expr:
        Converts the sigma subtree into an equivalent Adams polynomial.

    _to_adams_lambda(operands: set[Operand], max_adams_degree: int, as_symbol: bool = False, adams_degree: int = 1) -> sp.Expr:
        Converts the sigma subtree into an equivalent Adams polynomial, optimized when called
        from `to_lambda`.
    """

    def __repr__(self) -> str:
        """
        Returns the string representation of the sigma operator.

        Returns:
        --------
        str
            A string representation in the form "σ{degree}(child)".
        """
        return f"σ{self.degree}({self.child})"

    def _sympystr(self, printer: StrPrinter) -> str:
        """
        Returns the SymPy string representation of the sigma operator.

        Args:
        -----
        printer : StrPrinter
            The SymPy printer used to generate the string representation.

        Returns:
        --------
        str
            The string representation of the sigma operator in the form "σ{degree}(child)".
        """
        return f"σ{self.degree}({printer._print(self.child)})"

    def _latex(self, printer: StrPrinter) -> str:
        """
        Returns the LaTeX representation of the ring operator.

        Args:
        -----
        printer : StrPrinter
            The printer used to generate the LaTeX representation.

        Returns:
        --------
        str
            The LaTeX representation of the ring operator.
        """
        return f"σ^{{{self.degree}}}({printer._print(self.child)})"

    def get_max_groth_degree(self) -> int:
        """
        Computes the maximum sigma or lambda degree required for the Grothendieck context.

        The maximum degree is the higher of the sigma operator's degree or the maximum degree
        of the child node.

        Returns:
        --------
        int
            The maximum sigma or lambda degree required for the tree.
        """
        return max(self.degree, self.child.get_max_groth_degree())

    def _to_adams(
        self, operands: set[Operand], max_adams_degree: int, as_symbol: bool = False
    ) -> sp.Expr:
        """
        Converts the sigma subtree into an equivalent Adams polynomial.

        This method converts the sigma operator into its corresponding Adams polynomial
        by first converting the child to its Adams form, then applying Adams operators to
        the result. Finally, the Adams polynomials are substituted into the sigma to Adams
        conversion polynomial.

        Args:
        -----
        operands : set[Operand]
            The set of all operands in the expression tree.
        max_adams_degree : int
            The maximum Adams degree in the expression.
        as_symbol : bool, optional
            Whether to represent the Adams operators as symbols. Defaults to False.

        Returns:
        --------
        sp.Expr
            A polynomial of Adams operators equivalent to the sigma subtree.
        """
        if self.degree == 0:
            return sp.Integer(1)

        lrc = LambdaRingContext()

        # Get the polynomial by calling _to_adams on the child
        ph = self.child._to_adams(operands, max_adams_degree, as_symbol)
        # Create a list of the polynomial ph so that ph_list[j] = ψj(ph) for all j
        ph_list = [ph for _ in range(self.degree + 1)]

        # Apply the Adams operators to the polynomials in the list
        for j in range(2, self.degree + 1):
            for operand in operands:
                ph_list[j] = operand._apply_adams(
                    j, ph_list[j], max_adams_degree, as_symbol
                )

        adams_to_sigma = lrc.get_adams_2_sigma_pol(self.degree)

        # Substitute the jth polynomial for the Adams of degree j in the adams_to_sigma polynomial
        return adams_to_sigma.xreplace(
            {lrc.adams_vars[i]: ph_list[i] for i in range(self.degree + 1)}
        )

    def _to_adams_lambda(
        self,
        operands: set[Operand],
        max_adams_degree: int,
        as_symbol: bool = False,
        adams_degree: int = 1,
    ) -> sp.Expr:
        """
        Converts the sigma subtree into an equivalent Adams polynomial, optimized for lambda conversion.

        This method is similar to `_to_adams`, but it includes an optimization when the sigma operator
        is being converted directly to lambda in the `to_lambda` method. If the child is an operand and
        there are no Adams operators above, the sigma operator can be converted to lambda directly.

        Args:
        -----
        operands : set[Operand]
            The set of all operands in the expression tree.
        max_adams_degree : int
            The maximum Adams degree in the expression.
        as_symbol : bool, optional
            Whether to represent the Adams operators as symbols. Defaults to False.
        adams_degree : int, optional
            The cumulative Adams degree higher than this node in the expression tree. Defaults to 1.

        Returns:
        --------
        sp.Expr
            A polynomial of Adams operators equivalent to the sigma subtree.
        """
        if self.degree == 0:
            return sp.Integer(1)

        lrc = LambdaRingContext()

        # Optimization: If the child is an operand and there are no Adams operators on top,
        # convert sigma directly to lambda
        if isinstance(self.child, Operand) and adams_degree == 1:
            lambda_to_sigma = lrc.get_lambda_2_sigma_pol(self.degree)
            return lambda_to_sigma.xreplace(
                {
                    lrc.lambda_vars[i]: self.child.get_lambda_var(i, as_symbol)
                    for i in range(self.degree + 1)
                }
            )

        # Get the polynomial by calling _to_adams on the child
        ph = self.child._to_adams_lambda(
            operands, max_adams_degree, as_symbol, adams_degree + self.degree
        )

        if self.degree == 1:
            return ph

        # Create a list of the polynomial ph so that ph_list[j] = ψj(ph) for all j
        ph_list = [ph for _ in range(self.degree + 1)]

        # Apply the Adams operators to the polynomials in the list
        for j in range(2, self.degree + 1):
            for operand in operands:
                ph_list[j] = operand._apply_adams(
                    j, ph_list[j], max_adams_degree, as_symbol
                )

        adams_to_sigma = lrc.get_adams_2_sigma_pol(self.degree)

        # Substitute the jth polynomial for the Adams of degree j in the adams_to_sigma polynomial
        return adams_to_sigma.xreplace(
            {lrc.adams_vars[i]: ph_list[i] for i in range(self.degree + 1)}
        )


class Lambda_(RingOperator):
    """
    Represents the lambda ring operator in an expression tree.

    The `Lambda_` class is a specific type of `RingOperator` that applies the lambda operation
    to an expression. It should be created using `Lambda_(degree, child)`, where `degree`
    is the degree of the lambda operator, and `child` is the expression to which the lambda
    operator is applied.

    Methods:
    --------
    __repr__() -> str:
        Returns the string representation of the lambda operator.

    _sympystr(printer: StrPrinter) -> str:
        Returns the SymPy string representation of the lambda operator.

    get_max_groth_degree() -> int:
        Computes the maximum sigma or lambda degree needed to create a Grothendieck context
        for the expression tree.

    _to_adams(operands: set[Operand], max_adams_degree: int, as_symbol: bool = False) -> sp.Expr:
        Converts the lambda subtree into an equivalent Adams polynomial.

    _to_adams_lambda(operands: set[Operand], max_adams_degree: int, as_symbol: bool = False, adams_degree: int = 1) -> sp.Expr:
        Converts the lambda subtree into an equivalent Adams polynomial, optimized when called
        from `to_lambda`.
    """

    def _latex(self, printer: StrPrinter) -> str:
        """
        Returns the LaTeX representation of the ring operator.

        Args:
        -----
        printer : StrPrinter
            The printer used to generate the LaTeX representation.

        Returns:
        --------
        str
            The LaTeX representation of the ring operator.
        """
        return f"λ^{{{self.degree}}}({printer._print(self.child)})"

    def __repr__(self) -> str:
        """
        Returns the string representation of the lambda operator.

        Returns:
        --------
        str
            A string representation in the form "λ{degree}(child)".
        """
        return f"λ{self.degree}({self.child})"

    def _sympystr(self, printer: StrPrinter) -> str:
        """
        Returns the SymPy string representation of the lambda operator.

        Args:
        -----
        printer : StrPrinter
            The SymPy printer used to generate the string representation.

        Returns:
        --------
        str
            The string representation of the lambda operator in the form "λ{degree}(child)".
        """
        return f"λ{self.degree}({printer._print(self.child)})"

    def get_max_groth_degree(self) -> int:
        """
        Computes the maximum sigma or lambda degree required for the Grothendieck context.

        The maximum degree is the higher of the lambda operator's degree or the maximum degree
        of the child node.

        Returns:
        --------
        int
            The maximum sigma or lambda degree required for the tree.
        """
        return max(self.degree, self.child.get_max_groth_degree())

    def _to_adams(
        self, operands: set[Operand], max_adams_degree: int, as_symbol: bool = False
    ) -> sp.Expr:
        """
        Converts the lambda subtree into an equivalent Adams polynomial.

        This method converts the lambda operator into its corresponding Adams polynomial
        by first converting the child to its Adams form, then applying Adams operators to
        the result. Finally, the Adams polynomials are substituted into the lambda to Adams
        conversion polynomial.

        Args:
        -----
        operands : set[Operand]
            The set of all operands in the expression tree.
        max_adams_degree : int
            The maximum Adams degree in the expression.
        as_symbol : bool, optional
            Whether to represent the Adams operators as symbols. Defaults to False.

        Returns:
        --------
        sp.Expr
            A polynomial of Adams operators equivalent to the lambda subtree.
        """
        if self.degree == 0:
            return sp.Integer(1)

        lrc = LambdaRingContext()

        # Get the polynomial by calling _to_adams on the child
        ph = self.child._to_adams(operands, max_adams_degree, as_symbol)
        # Create a list of the polynomial ph so that ph_list[j] = ψj(ph) for all j
        ph_list = [ph for _ in range(self.degree + 1)]

        # Apply the Adams operators to the polynomials in the list
        for j in range(2, self.degree + 1):
            for operand in operands:
                ph_list[j] = operand._apply_adams(
                    j, ph_list[j], max_adams_degree, as_symbol
                )

        adams_to_lambda = lrc.get_adams_2_lambda_pol(self.degree)

        # Substitute the jth polynomial for the Adams of degree j in the adams_to_lambda polynomial
        return adams_to_lambda.xreplace(
            {lrc.adams_vars[i]: ph_list[i] for i in range(self.degree + 1)}
        )

    def _to_adams_lambda(
        self,
        operands: set[Operand],
        max_adams_degree: int,
        as_symbol: bool = False,
        adams_degree: int = 1,
    ) -> sp.Expr:
        """
        Converts the lambda subtree into an equivalent Adams polynomial, optimized for lambda conversion.

        This method is similar to `_to_adams`, but it includes an optimization when the lambda operator
        is being converted directly to lambda in the `to_lambda` method. If the child is an operand and
        there are no Adams operators above, the lambda operator can be converted directly.

        Args:
        -----
        operands : set[Operand]
            The set of all operands in the expression tree.
        max_adams_degree : int
            The maximum Adams degree in the expression.
        as_symbol : bool, optional
            Whether to represent the Adams operators as symbols. Defaults to False.
        adams_degree : int, optional
            The sum of the degree of all Adams operators higher than this node in the expression tree.
            Defaults to 1.

        Returns:
        --------
        sp.Expr
            A polynomial of Adams operators equivalent to the lambda subtree.
        """
        if self.degree == 0:
            return sp.Integer(1)

        lrc = LambdaRingContext()

        # Optimization: If the child is an operand and there are no Adams operators on top,
        # return the lambda variable directly
        if isinstance(self.child, Operand) and adams_degree == 1:
            return self.child.get_lambda_var(self.degree, as_symbol)

        # Get the polynomial by calling _to_adams_lambda on the child
        ph = self.child._to_adams_lambda(
            operands, max_adams_degree, as_symbol, adams_degree + self.degree
        )

        if self.degree == 1:
            return ph

        # Create a list of the polynomial ph so that ph_list[j] = ψj(ph) for all j
        ph_list = [ph for _ in range(self.degree + 1)]

        # Apply the Adams operators to the polynomials in the list
        for j in range(2, self.degree + 1):
            for operand in operands:
                ph_list[j] = operand._apply_adams(
                    j, ph_list[j], max_adams_degree, as_symbol
                )

        adams_to_lambda = lrc.get_adams_2_lambda_pol(self.degree)

        # Substitute the jth polynomial for the Adams of degree j in the adams_to_lambda polynomial
        return adams_to_lambda.xreplace(
            {lrc.adams_vars[i]: ph_list[i] for i in range(self.degree + 1)}
        )


class Adams(RingOperator):
    """
    Represents the Adams ring operator in an expression tree.

    The `Adams` class is a specific type of `RingOperator` that applies the Adams operation
    to an expression. It should be created using `Adams(degree, child)`, where `degree`
    is the degree of the Adams operator, and `child` is the expression to which the Adams
    operator is applied.

    Methods:
    --------
    __repr__() -> str:
        Returns the string representation of the Adams operator.

    _sympystr(printer: StrPrinter) -> str:
        Returns the SymPy string representation of the Adams operator.

    get_max_groth_degree() -> int:
        Computes the maximum sigma or lambda degree needed to create a Grothendieck context
        for the expression tree.

    _to_adams(operands: set[Operand], max_adams_degree: int, as_symbol: bool = False) -> sp.Expr:
        Converts the Adams subtree into an equivalent Adams polynomial.

    _to_adams_lambda(operands: set[Operand], max_adams_degree: int, as_symbol: bool = False, adams_degree: int = 1) -> sp.Expr:
        Converts the Adams subtree into an equivalent Adams polynomial, optimized when called
        from `to_lambda`.
    """

    def _latex(self, printer: StrPrinter) -> str:
        """
        Returns the LaTeX representation of the ring operator.

        Args:
        -----
        printer : StrPrinter
            The printer used to generate the LaTeX representation.

        Returns:
        --------
        str
            The LaTeX representation of the ring operator.
        """
        return f"ψ^{{{self.degree}}}({printer._print(self.child)})"

    def __repr__(self) -> str:
        """
        Returns the string representation of the Adams operator.

        Returns:
        --------
        str
            A string representation in the form "ψ{degree}(child)".
        """
        return f"ψ{self.degree}({self.child})"

    def _sympystr(self, printer: StrPrinter) -> str:
        """
        Returns the SymPy string representation of the Adams operator.

        Args:
        -----
        printer : StrPrinter
            The SymPy printer used to generate the string representation.

        Returns:
        --------
        str
            The string representation of the Adams operator in the form "ψ{degree}(child)".
        """
        return f"ψ{self.degree}({printer._print(self.child)})"

    def get_max_groth_degree(self) -> int:
        """
        Computes the maximum sigma or lambda degree required for the Grothendieck context.

        Since the Adams operator does not change the Grothendieck degree, this method
        returns the maximum Grothendieck degree of the child node.

        Returns:
        --------
        int
            The maximum sigma or lambda degree required for the tree.
        """
        return self.child.get_max_groth_degree()

    def _to_adams(
        self, operands: set[Operand], max_adams_degree: int, as_symbol: bool = False
    ) -> sp.Expr:
        """
        Converts the Adams subtree into an equivalent Adams polynomial.

        This method converts the child expression into its Adams form, then applies the
        Adams operator to the result. If the degree is 1, it directly returns the child.
        For higher degrees, it applies the Adams operator to all operands in the expression.

        Args:
        -----
        operands : set[Operand]
            The set of all operands in the expression tree.
        max_adams_degree : int
            The maximum Adams degree in the expression.
        as_symbol : bool, optional
            Whether to represent the Adams operators as symbols. Defaults to False.

        Returns:
        --------
        sp.Expr
            A polynomial of Adams operators equivalent to the Adams subtree.
        """
        if self.degree == 0:
            return sp.Integer(1)

        # Get the polynomial by calling _to_adams on the child
        ph = self.child._to_adams(operands, max_adams_degree, as_symbol)

        if self.degree == 1:
            return ph

        # Apply the Adams operator to all operands
        for operand in operands:
            ph = operand._apply_adams(self.degree, ph, max_adams_degree, as_symbol)

        return ph

    def _to_adams_lambda(
        self,
        operands: set[Operand],
        max_adams_degree: int,
        as_symbol: bool = False,
        adams_degree: int = 1,
    ) -> sp.Expr:
        """
        Converts the Adams subtree into an equivalent Adams polynomial, optimized for lambda conversion.

        This method is similar to `_to_adams`, but it includes an optimization when the Adams operator
        is being converted directly to lambda in the `to_lambda` method. The child expression is first
        converted to its Adams form, and if the degree is higher than 1, the Adams operator is applied
        to all operands in the expression.

        Args:
        -----
        operands : set[Operand]
            The set of all operands in the expression tree.
        max_adams_degree : int
            The maximum Adams degree in the expression.
        as_symbol : bool, optional
            Whether to represent the Adams operators as symbols. Defaults to False.
        adams_degree : int, optional
            The sum of the degree of all Adams operators higher than this node in the expression tree.
            Defaults to 1.

        Returns:
        --------
        sp.Expr
            A polynomial of Adams operators equivalent to the Adams subtree.
        """
        if self.degree == 0:
            return sp.Integer(1)

        # Get the polynomial by calling _to_adams_lambda on the child
        ph = self.child._to_adams_lambda(
            operands, max_adams_degree, as_symbol, adams_degree + self.degree
        )

        if self.degree == 1:
            return ph

        # Apply the Adams operator to all operands
        for operand in operands:
            ph = operand._apply_adams(self.degree, ph, max_adams_degree, as_symbol)

        return ph


def to_adams(self: sp.Expr, as_symbol: bool = False) -> sp.Expr:
    """
    Converts the current expression into a polynomial of Adams operators.

    This function transforms the expression into its equivalent form using Adams operators.
    It is intended to be used as a method for `Pow`, `Add`, `Mul`, and `Rational` expressions.
    The Adams polynomial represents the structure of the current expression in the context of
    Grothendieck rings.

    Args:
    -----
    as_symbol : bool, optional
        Whether to represent the Adams operators as symbols. Defaults to False.

    Returns:
    --------
    sp.Expr
        The polynomial of Adams operators equivalent to the current expression.
    """
    operands: set[Operand] = self.free_symbols
    max_adams_degree = self.get_max_adams_degree()

    return self._to_adams(
        operands, max_adams_degree=max_adams_degree, as_symbol=as_symbol
    )


def to_lambda(
    self: sp.Expr, as_symbol: bool = False, *, optimize: bool = True
) -> sp.Expr:
    """
    Converts the current expression into a polynomial of lambda operators.

    This function transforms the expression into its equivalent form using lambda operators.
    It is intended to be used as a method for `Pow`, `Add`, `Mul`, and `Rational` expressions.
    The conversion is done via Adams operators, which are then substituted by their lambda equivalents.

    Args:
    -----
    as_symbol : bool, optional
        Whether to represent the Adams operators as symbols. Defaults to False.
    optimize : bool, optional, default=True
        Whether to optimize the conversion to lambda. If True, the conversion uses an optimized
        pathway by converting Adams operators directly to lambda operators.

    Returns:
    --------
    sp.Expr
        The polynomial of lambda operators equivalent to the current expression.
    """
    operands: set[Operand] = self.free_symbols
    max_adams_degree = self.get_max_adams_degree()

    # Get the Adams polynomial of the tree, with optimization if requested
    if optimize:
        adams_pol = self._to_adams_lambda(
            operands,
            max_adams_degree=max_adams_degree,
            as_symbol=as_symbol,
            adams_degree=1,
        )
    else:
        adams_pol = self._to_adams(
            operands, max_adams_degree=max_adams_degree, as_symbol=as_symbol
        )

    # If the result is an integer, return it directly
    if isinstance(adams_pol, (sp.Integer, int)):
        return adams_pol

    # Substitute Adams variables for lambda variables
    for operand in operands:
        adams_pol = operand._subs_adams(adams_pol, max_adams_degree, as_symbol)

    return adams_pol


@typechecked
def sigma(self: sp.Expr, degree: int) -> sp.Expr:
    """
    Applies the sigma operation to the current expression.

    This function adds the sigma ring operator to the expression, creating a new expression
    with the specified sigma operator applied. It is intended to be used as a method for
    `Pow`, `Add`, `Mul`, and `Rational` expressions.

    Args:
    -----
    degree : int
        The degree of the sigma operator to be applied.

    Returns:
    --------
    sp.Expr
        An expression with the sigma operator applied.
    """
    return Sigma(degree, self)


@typechecked
def lambda_(self: sp.Expr, degree: int) -> sp.Expr:
    """
    Applies the lambda operation to the current expression.

    This function adds the lambda ring operator to the expression, creating a new expression
    with the specified lambda operator applied. It is intended to be used as a method for
    `Pow`, `Add`, `Mul`, and `Rational` expressions.

    Args:
    -----
    degree : int
        The degree of the lambda operator to be applied.

    Returns:
    --------
    sp.Expr
        An expression with the lambda operator applied.
    """
    if degree == 0:
        return sp.Integer(1)
    if degree == 1:
        return self
    return Lambda_(degree, self)


@typechecked
def adams(self: sp.Expr, degree: int) -> sp.Expr:
    """
    Applies the Adams operation to the current expression.

    This function adds the Adams ring operator to the expression, creating a new expression
    with the specified Adams operator applied. It is intended to be used as a method for
    `Pow`, `Add`, `Mul`, and `Rational` expressions.

    Args:
    -----
    degree : int
        The degree of the Adams operator to be applied.

    Returns:
    --------
    sp.Expr
        An expression with the Adams operator applied.
    """
    if degree == 0:
        return 1
    if degree == 1:
        return self
    return Adams(degree, self)
