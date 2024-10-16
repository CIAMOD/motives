# node.py

from __future__ import annotations  # For forward references in type hints
from typing import TypeVar
import sympy as sp
from typeguard import typechecked

from .lambda_ring_context import LambdaRingContext

Operand = TypeVar('Operand')  # Define Operand as a TypeVar for type hinting

class LambdaRingExpr(sp.Expr):
    """
    Represents a mathematical expression that can be converted to different
    polynomial forms such as Adams and Lambda polynomials, or manipulated using
    various operations like sigma, lambda, and adams. 

    Subclasses must implement methods to handle specific types of expressions.
    """

    def get_max_adams_degree(self) -> int:
        """
        Computes the maximum Adams degree of this expression.

        This method must be implemented in subclasses.

        Returns:
        --------
        int
            The maximum adams degree of this tree once converted to an adams polynomial.

        Raises:
        -------
        NotImplementedError
            If this method is not implemented in the subclass.
        """
        raise NotImplementedError("This method must be implemented in all subclasses.")

    def get_max_groth_degree(self) -> int:
        """
        Computes the maximum degree needed to create a Grothendieck context for this expression.

        This method calculates the maximum sigma or lambda degree required for this expression tree.

        Returns:
        --------
        int
            The maximum sigma or lambda degree of the expression tree.

        Raises:
        -------
        NotImplementedError
            If this method is not implemented in the subclass.
        """
        raise NotImplementedError("This method must be implemented in all subclasses.")

    def to_adams(self, lrc: LambdaRingContext = None) -> sp.Expr:
        """
        Converts this expression into an equivalent Adams polynomial.

        Args:
        -----
        lrc : LambdaRingContext, optional
            The Grothendieck ring context used for converting between ring operators.
            If None, a new context is created.

        Returns:
        --------
        sp.Expr
            A polynomial of Adams operators equivalent to this expression.
        """
        lrc = lrc or LambdaRingContext()
        operands: set[Operand] = self.free_symbols
        return self._to_adams(operands, lrc)

    def to_lambda(self, lrc: LambdaRingContext = None, *, optimize: bool = True) -> sp.Expr:
        """
        Converts this expression into an equivalent Lambda polynomial.

        Args:
        -----
        lrc : LambdaRingContext, optional
            The Grothendieck ring context used for the conversion between ring operators.
            If None, a new context is created.
        optimize : bool, optional
            If True, optimizations are applied during the conversion.

        Returns:
        --------
        sp.Expr
            A polynomial of Lambda operators equivalent to this expression.
        """
        lrc = lrc or LambdaRingContext()
        operands: set[Operand] = self.free_symbols

        if optimize:
            adams_pol = self._to_adams_lambda(operands, lrc, 1)
        else:
            adams_pol = self._to_adams(operands, lrc)

        for operand in operands:
            adams_pol = operand._subs_adams(lrc, adams_pol)

        return adams_pol

    @typechecked
    def sigma(self, degree: int) -> sp.Expr:
        """
        Applies the sigma operation to this expression.

        Args:
        -----
        degree : int
            The degree of the sigma operator to apply.

        Returns:
        --------
        sp.Expr
            The expression with the sigma operator applied.
        """
        from .operator.ring_operator import Sigma
        return Sigma(degree, self)

    @typechecked
    def lambda_(self, degree: int) -> sp.Expr:
        """
        Applies the lambda operation to this expression.

        Args:
        -----
        degree : int
            The degree of the lambda operator to apply.

        Returns:
        --------
        sp.Expr
            The expression with the lambda operator applied.
        """
        from .operator.ring_operator import Lambda_
        return Lambda_(degree, self)

    @typechecked
    def adams(self, degree: int) -> sp.Expr:
        """
        Applies the Adams operation to this expression.

        Args:
        -----
        degree : int
            The degree of the Adams operator to apply.

        Returns:
        --------
        sp.Expr
            The expression with the Adams operator applied.
        """
        from .operator.ring_operator import Adams
        return Adams(degree, self)

    def _to_adams(self, operands: set[Operand], lrc: LambdaRingContext) -> sp.Expr:
        """
        Converts this expression subtree into an equivalent Adams polynomial.

        Args:
        -----
        operands : set[Operand]
            The set of all operands in the expression tree.
        lrc : LambdaRingContext
            The Grothendieck ring context used for the conversion between ring operators.

        Returns:
        --------
        sp.Expr
            A polynomial of Adams operators equivalent to this expression subtree.

        Raises:
        -------
        NotImplementedError
            If this method is not implemented in the subclass.
        """
        raise NotImplementedError("This method must be implemented in all subclasses.")

    def _to_adams_lambda(self, operands: set[Operand], lrc: LambdaRingContext, adams_degree: int = 1) -> sp.Expr:
        """
        Converts this expression subtree into an equivalent Adams polynomial, with optimizations for lambda conversion.

        This method is used when the `optimize` flag is set to True in the `to_lambda` method.

        Args:
        -----
        operands : set[Operand]
            The set of all operands in the expression tree.
        lrc : LambdaRingContext
            The Grothendieck ring context used for the conversion between ring operators.
        adams_degree : int, optional
            Sum of the degree of all Adams operators higher than this node in its branch.

        Returns:
        --------
        sp.Expr
            A polynomial of Adams operators equivalent to this expression subtree.

        Raises:
        -------
        NotImplementedError
            If this method is not implemented in the subclass.
        """
        raise NotImplementedError("This method must be implemented in all subclasses.")
