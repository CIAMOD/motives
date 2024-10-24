import sympy as sp
from multipledispatch import dispatch

from ...core import LambdaRingContext
from ...core.operand import Operand

from ..motive import Motive
from ..lefschetz import Lefschetz


class GL(Motive, sp.AtomicExpr):
    """
    Represents a GL_n bundle in an expression tree.

    A GL_n bundle represents the product \prod_{k=0}^{n-1} (L^n - L^k), where L is the Lefschetz motive.
    This class supports operations related to Adams and Lambda transformations.

    Attributes:
    -----------
    n : int
        The dimension of the GL_n bundle.
    lef : Lefschetz
        The Lefschetz motive associated with the bundle.
    _et_repr : sp.Expr
        The GL_n bundle as a sympy expression.
    _lambda_vars : dict[int, sp.Expr]
        A dictionary of the lambda variables generated for this GL_n bundle.
    """

    def __new__(cls, n: int, *args, **kwargs):
        """
        Creates a new instance of `GL`.

        Args:
        -----
        n : int
            The dimension of the GL_n bundle.

        Returns:
        --------
        GL
            A new instance of the `GL` class.
        """
        new_gl = sp.AtomicExpr.__new__(cls)
        new_gl._assumptions["commutative"] = True
        return new_gl

    def __init__(self, n: int, *args, **kwargs) -> None:
        """
        Initializes a `GL` instance.

        Args:
        -----
        n : int
            The dimension of the GL_n bundle.
        """
        self.n: int = n
        self.lef: Lefschetz = Lefschetz()
        self._et_repr: sp.Expr = self.lef ** (
            ((self.n - 1) * (self.n - 2)) // 2
        ) * sp.Mul(*[self.lef**k - 1 for k in range(1, self.n + 1)])
        self._lambda_vars: dict[int, sp.Expr] = {}

    def __repr__(self) -> str:
        """
        Returns the string representation of the GL_n bundle.

        Returns:
        --------
        str
            A string representation in the form "GL_n".
        """
        return f"GL_{self.n}"

    def _hashable_content(self) -> tuple:
        """
        Returns the hashable content of the GL_n bundle.

        Returns:
        --------
        tuple
            A tuple containing the dimension of the GL_n bundle.
        """
        return (self.n,)

    def get_adams_var(self, i: int) -> sp.Expr:
        """
        Returns the GL_n bundle with an Adams operation applied to it.

        The Adams operation on the GL_n bundle is equivalent to applying the Adams operator
        to each term in the product formula for GL_n.

        Args:
        -----
        i : int
            The degree of the Adams operator.

        Returns:
        --------
        sp.Expr
            The GL_n bundle with the Adams operator applied.
        """
        return self.lef ** (((self.n - 1) * (self.n - 2)) // 2 * i) * sp.Mul(
            *[self.lef ** (i * k) - 1 for k in range(1, self.n + 1)]
        )

    def get_lambda_var(self, i: int) -> sp.Expr:
        """
        Returns the GL_n bundle with a Lambda operation applied to it.

        The Lambda operation is applied by converting Adams variables into Lambda variables
        using the context's transformation rules.

        Args:
        -----
        i : int
            The degree of the Lambda operator.

        Returns:
        --------
        sp.Expr
            The GL_n bundle with the Lambda operator applied.
        """
        if i not in self._lambda_vars:
            lrc = LambdaRingContext()

            ph_list = [self.lef._to_adams(j, self._et_repr) for j in range(i + 1)]

            self._lambda_vars[i] = lrc.get_adams_2_lambda_pol(i).xreplace(
                {lrc.adams_vars[i]: ph_list[i] for i in range(i + 1)}
            )

        return self._lambda_vars[i]

    @dispatch(int, sp.Expr)
    def _to_adams(self, degree: int, ph: sp.Expr) -> sp.Expr:
        """
        Applies the Adams operator to any instances of this GL_n bundle in the expression.

        This method raises an exception because GL_n bundles should not appear directly
        in the expression. Instead, they should be decomposed into their components.

        Args:
        -----
        degree : int
            The degree of the Adams operator to apply.
        ph : sp.Expr
            The polynomial in which to apply the Adams operator.

        Raises:
        -------
        Exception
            Always raised as GL_n bundles should be decomposed into their components.
        """
        raise Exception(
            f"There is a motive in the expression {ph}. "
            "It should have been converted to its components."
        )

    @dispatch(set)
    def _to_adams(self, operands: set[Operand]) -> sp.Expr:
        """
        Converts this GL_n bundle into an equivalent Adams polynomial.

        Args:
        -----
        operands : set[Operand]
            The set of all operands in the expression tree.

        Returns:
        --------
        sp.Expr
            The Adams polynomial equivalent to this GL_n bundle.
        """
        return self.lef ** (((self.n - 1) * (self.n - 2)) // 2) * sp.Mul(
            *[self.lef**k - 1 for k in range(1, self.n + 1)]
        )

    def _subs_adams(self, ph: sp.Expr) -> sp.Expr:
        """
        Substitutes Adams variables of this GL_n bundle in the expression
        with their equivalent Lambda polynomials.

        This method raises an exception because GL_n bundles should not appear
        directly in the expression. They should be decomposed into their components.

        Args:
        -----
        ph : sp.Expr
            The polynomial in which to substitute the Adams variables.

        Raises:
        -------
        Exception
            Always raised as GL_n bundles should be decomposed into their components.
        """
        raise Exception(
            f"There is a GL in the expression {ph}. "
            "It should have been converted to its components."
        )

    @property
    def free_symbols(self) -> set[sp.Symbol]:
        """
        Returns the set of free symbols in the GL_n bundle, which includes
        the Lefschetz motive.

        Returns:
        --------
        set[sp.Symbol]
            The set of free symbols in the GL_n bundle.
        """
        return {self.lef}
