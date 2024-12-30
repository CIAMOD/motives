# node.py

from __future__ import annotations  # For forward references in type hints
from typing import TypeVar
import sympy as sp
from typeguard import typechecked


Operand = TypeVar("Operand")  # Define Operand as a TypeVar for type hinting


class LambdaRingExpr(sp.Expr):
    """
    Represents a mathematical symbolic expression in some lambda-ring R. The folloing is assumed.
        - R is a unital abelian ring.
        - The ring R has two opposite lambda-ring structures λ and σ.
        - The lambda-structure σ is special.
        - σ has associated Adams operations, denoted by ψ.
    It can be converted to different polynomial forms such as Adams and Lambda polynomials,
    or manipulated using various operations like sigma, lambda, and Adams.

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

    def to_adams(self) -> sp.Expr:
        """
        Converts this expression into an equivalent Adams polynomial, a polynomial
        depending only on Adams operations of the leaves of the given expression Tree.

        Returns:
        --------
        sp.Expr
            A polynomial of Adams operators equivalent to this expression.
        """
        operands: set[Operand] = self.free_symbols
        return self._to_adams(operands)

    def to_lambda(self, *, optimize: bool = True) -> sp.Expr:
        """
        Converts this expression into an equivalent lambda polynomial, a polynomial
        depending only on lambda operations of the leaves of the given expression Tree.

        Args:
        -----
        optimize : bool, optional
            If True, optimizations are applied during the conversion.

        Returns:
        --------
        sp.Expr
            A polynomial of Lambda operators equivalent to this expression.
        """
        operands: set[Operand] = self.free_symbols

        if optimize:
            adams_pol = self._to_adams_lambda(operands, 1)
        else:
            adams_pol = self._to_adams(operands)

        for operand in operands:
            adams_pol = operand._subs_adams(adams_pol)

        return adams_pol

    @typechecked
    def sigma(self, degree: int) -> sp.Expr:
        """
        Applies the sigma operation of order n to this expression.

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
        Applies the lambda operation of order n to this expression.

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
        Applies the Adams operation of order n to this expression.

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

    def _to_adams(self, operands: set[Operand]) -> sp.Expr:
        """
        Converts this expression subtree into an equivalent Adams polynomial.

        Args:
        -----
        operands : set[Operand]
            The set of all operands in the expression tree.

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

    def _to_adams_lambda(
        self, operands: set[Operand], adams_degree: int = 1
    ) -> sp.Expr:
        """
        Converts this expression subtree into an equivalent Adams polynomial, with optimizations for lambda conversion, 
        converting only lambda operations into Adams polynomials if needed and letting subexpressions which are already
        lambda polynomials unnafected if no adams, lambda or sigma operation acts on them.

        This method is used when the `optimize` flag is set to True in the `to_lambda` method.

        Args:
        -----
        operands : set[Operand]
            The set of all operands in the expression tree.
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
