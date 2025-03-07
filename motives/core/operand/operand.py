# operand.py

from __future__ import annotations  # For forward references
from functools import reduce
import math
import sympy as sp
from sympy.printing.str import StrPrinter

from ..lambda_ring_expr import LambdaRingExpr


class Operand(LambdaRingExpr):
    """
    An abstract node in an expression that represents an operand.

    This class serves as a base class for any new operand in the expression tree.
    All operands in an expression must be subclasses of this class and implement
    specific behavior for conversion to Adams and lambda polynomials, as well as other operations.
    """

    def _sympystr(self, printer: StrPrinter) -> str:
        """
        Provides a string representation of the operand for SymPy printing.

        Args:
        -----
        printer : StrPrinter
            The SymPy printer used for formatting the string representation.

        Returns:
        --------
        str
            The string representation of the operand.
        """
        return self.__repr__()

    def _latex(self, printer: StrPrinter) -> str:
        """
        Provides a LaTeX representation of the operand for SymPy printing.

        Args:
        -----
        printer : StrPrinter
            The SymPy printer used for formatting the LaTeX representation.

        Returns:
        --------
        str
            The LaTeX representation of the operand.
        """
        return self.__repr__()

    def get_max_adams_degree(self) -> int:
        """
        Computes the maximum Adams degree of this operand.

        This method multiplies the Adams, Lambda, and Sigma degrees across every branch
        of the tree and returns the maximum value. For an individual operand in a leaf of the tree,
        this value is typically `1`.

        Returns:
        --------
        int
            The maximum Adams degree of this operand when converted to an Adams polynomial.
        """
        return 1

    def get_max_groth_degree(self) -> int:
        """
        Computes the maximum operator degree needed to create a Grothendieck context for this operand.

        Returns:
        --------
        int
            The maximum sigma or lambda degree required for this operand in the expression tree.
        """
        return 0

    def get_adams_var(self, i: int, as_symbol: bool = False) -> sp.Expr:
        """
        Returns the operand with an Adams operation of order i applied to it.

        Args:
        -----
        i : int
            The degree of the Adams operator to apply.
        as_symbol : bool, optional
            If True, returns the Adams variable as a SymPy Symbol. Otherwise, returns it as an
            Adams_ object.

        Returns:
        --------
        sp.Expr
            The operand with the Adams operator applied.

        Raises:
        -------
        NotImplementedError
            If this method is not implemented in the subclass.
        """
        raise NotImplementedError("This method must be implemented in all subclasses.")

    def get_lambda_var(self, i: int, as_symbol: bool = False) -> sp.Expr:
        """
        Returns the operand with a lambda operation of order i applied to it.

        Args:
        -----
        i : int
            The degree of the Lambda operator to apply.
        as_symbol : bool, optional
            If True, returns the lambda variable as a SymPy Symbol. Otherwise, returns it as a
            Lambda_ object.

        Returns:
        --------
        sp.Expr
            The operand with the Lambda operator applied.

        Raises:
        -------
        NotImplementedError
            If this method is not implemented in the subclass.
        """
        raise NotImplementedError("This method must be implemented in all subclasses.")

    def _apply_adams(
        self, degree: int, ph: sp.Expr, max_adams_degree: int, as_symbol: bool = False
    ) -> sp.Expr:
        """
        Applies the Adams operation of the requested degree to instances of this operand and any of
        its Adams operations which appear in the polynomial ph.

        If ph=x._apply_adams(i,ph) is called iteratively for each operand x such that either x or some of
        its Adams operations appear in the polynomial expression P, then the resulting polynomial
        will represent ψ^i(ph).

        If neither x nor any of its Adams operations appear in ph, then ph is returned unaffected.

        Args:
        -----
        degree : int
            The degree of the Adams operator to apply.
        ph : sp.Expr
            The polynomial in which the Adams operator will be applied.
        max_adams_degree : int
            The maximum Adams degree in the expression.
        as_symbol : bool, optional
            If True, returns the result as a SymPy Symbol. Otherwise, returns it as an Adams_ object.

        Returns:
        --------
        sp.Expr
            The polynomial with the Adams operator applied.

        Raises:
        -------
        NotImplementedError
            If this method is not implemented in the subclass.
        """
        raise NotImplementedError("This method must be implemented in all subclasses.")

    def _to_adams(
        self, operands: set[Operand], max_adams_degree: int, as_symbol: bool = False
    ) -> sp.Expr:
        """
        Converts this operand into an equivalent Adams polynomial.

        For an operand, this simply returns the Adams polynomial of degree 1 for itself.

        Args:
        -----
        operands : set[Operand]
            The set of all operands in the expression tree.
        max_adams_degree : int
            The maximum Adams degree in the context of the conversion.
        as_symbol : bool, optional
            Whether to represent the Adams operators as symbols. Defaults to False.

        Returns:
        --------
        sp.Expr
            The Adams polynomial of degree 1 for this operand.
        """
        return self.get_adams_var(1, as_symbol)

    def _to_adams_lambda(
        self,
        operands: set[Operand],
        max_adams_degree: int,
        as_symbol: bool = False,
        adams_degree: int = 1,
    ) -> sp.Expr:
        """
        Converts this operand into an equivalent Adams polynomial, with optimizations for Lambda conversion.

        This method is called when `optimize=True` in the `to_lambda` method. For an operand,
        this simply returns the Adams polynomial of degree 1 for itself.

        Args:
        -----
        operands : set[Operand]
            The set of all operands in the expression tree.
        max_adams_degree : int
            The maximum Adams degree in the context of the conversion.
        as_symbol : bool, optional
            Whether to represent the Adams operators as symbols. Defaults to False.
        adams_degree : int, optional
            The sum of the degrees of all Adams operators higher than this node in its branch.

        Returns:
        --------
        sp.Expr
            The Adams polynomial of degree 1 for this operand.
        """
        return self._to_adams(operands, max_adams_degree, as_symbol)

    def _subs_adams(
        self, ph: sp.Expr, max_adams_degree: int, as_symbol: bool = False
    ) -> sp.Expr:
        """
        Substitutes any Adams operations on this operand in the given polynomial with their equivalent lambda polynomial.

        This method is used in `to_lambda` to convert any Adams operations in the polynomial
        into lambda operations.

        Args:
        -----
        ph : sp.Expr
            The polynomial in which to substitute the Adams operations.
        max_adams_degree : int
            The maximum Adams degree in the context of the substitution.
        as_symbol : bool, optional
            Whether to represent the Adams operators as symbols. Defaults to False.

        Returns:
        -------
        sp.Expr
            The polynomial with Adams operations substituted.

        Raises:
        -------
        NotImplementedError
            If this method is not implemented in the subclass.
        """
        raise NotImplementedError("This method must be implemented in all subclasses.")

    @property
    def free_symbols(self) -> set[Operand]:
        """
        Returns the set of all operands (free symbols) in the expression tree.

        Returns:
        --------
        set[Operand]
            A set containing the operands in this expression.
        """
        return {self}
