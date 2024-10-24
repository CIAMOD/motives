# operand.py

from __future__ import annotations  # For forward references
from typing import TypeVar
import sympy as sp
from sympy.printing.str import StrPrinter
from multipledispatch import dispatch

from .lambda_ring_expr import LambdaRingExpr


class Operand(LambdaRingExpr):
    """
    An abstract node in an expression that represents an operand.

    This class serves as a base class for any new operand in the expression tree.
    All operands in an expression must be subclasses of this class and implement
    specific behavior for conversion to Adams and Lambda polynomials, as well as other operations.
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

    def get_max_adams_degree(self) -> int:
        """
        Computes the maximum Adams degree of this operand.

        This method multiplies the Adams, Lambda, and Sigma degrees across every branch
        of the tree and returns the maximum value. For an individual operand, this value
        is typically `1`.

        Returns:
        --------
        int
            The maximum Adams degree of this operand when converted to an Adams polynomial.
        """
        return 1

    def get_max_groth_degree(self) -> int:
        """
        Computes the maximum Grothendieck degree needed to create a context for this operand.

        Returns:
        --------
        int
            The maximum sigma or lambda degree required for this operand in the expression tree.
        """
        return 0

    def get_adams_var(self, i: int) -> sp.Expr:
        """
        Returns the operand with an Adams operation applied to it.

        Args:
        -----
        i : int
            The degree of the Adams operator to apply.

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

    def get_lambda_var(self, i: int) -> sp.Expr:
        """
        Returns the operand with a Lambda operation applied to it.

        Args:
        -----
        i : int
            The degree of the Lambda operator to apply.

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

    @dispatch(int, sp.Expr)
    def _to_adams(self, degree: int, ph: sp.Expr) -> sp.Expr:
        """
        Applies the Adams operation to instances of this operand in a given polynomial.

        Args:
        -----
        degree : int
            The degree of the Adams operator to apply.
        ph : sp.Expr
            The polynomial in which the Adams operator will be applied.

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

    @dispatch(set)
    def _to_adams(self, operands: set[Operand]) -> sp.Expr:
        """
        Converts this operand into an equivalent Adams polynomial.

        For an operand, this simply returns the Adams polynomial of degree 1 for itself.

        Args:
        -----
        operands : set[Operand]
            The set of all operands in the expression tree.

        Returns:
        --------
        sp.Expr
            The Adams polynomial of degree 1 for this operand.

        Raises:
        -------
        NotImplementedError
            If this method is not implemented in the subclass.
        """
        raise NotImplementedError("This method must be implemented in all subclasses.")

    def _to_adams_lambda(
        self, operands: set[Operand], adams_degree: int = 1
    ) -> sp.Expr:
        """
        Converts this operand into an equivalent Adams polynomial, with optimizations for Lambda conversion.

        This method is called when `optimize=True` in the `to_lambda` method. For an operand,
        this simply returns the Adams polynomial of degree 1 for itself.

        Args:
        -----
        operands : set[Operand]
            The set of all operands in the expression tree.
        adams_degree : int, optional
            The sum of the degrees of all Adams operators higher than this node in its branch.

        Returns:
        --------
        sp.Expr
            The Adams polynomial of degree 1 for this operand.
        """
        return self._to_adams(operands)

    def _subs_adams(self, ph: sp.Expr) -> sp.Expr:
        """
        Substitutes any Adams operations on this operand in the given polynomial with their equivalent Lambda polynomial.

        This method is used in `to_lambda` to convert any Adams operations in the polynomial
        into Lambda operations.

        Args:
        -----
        ph : sp.Expr
            The polynomial in which to substitute the Adams operations.

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
