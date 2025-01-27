# node.py

from __future__ import annotations  # For forward references in type hints
from typing import TypeVar, Set
import sympy as sp
from typeguard import typechecked

Operand = TypeVar("Operand")  # Define Operand as a TypeVar for type hinting


class LambdaRingExpr(sp.Expr):
    """
    Represents a mathematical symbolic expression in a lambda-ring R. The following assumptions are made:
        - R is a unital abelian ring.
        - The ring R has two opposite lambda-ring structures λ and σ.
        - The lambda-structure σ is special.
        - σ has associated Adams operations, denoted by ψ.

    This class supports conversion to different polynomial forms such as Adams and Lambda polynomials,
    and can be manipulated using various operations like sigma, lambda, and Adams.

    Subclasses must implement methods to handle specific types of expressions.
    """

    def get_max_adams_degree(self) -> int:
        """
        Computes the maximum Adams degree of this expression.

        This method must be implemented in subclasses.

        Returns:
        --------
        int
            The maximum Adams degree of the expression once converted to an Adams polynomial.

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

    def to_adams(self, as_symbol: bool = False) -> sp.Expr:
        """
        Converts this expression into an equivalent Adams polynomial, a polynomial
        depending only on Adams operations of the leaves of the given expression tree.

        Returns:
        --------
        sp.Expr
            A polynomial of Adams operators equivalent to this expression.
        """
        operands: Set[Operand] = self.free_symbols
        max_adams_degree = self.get_max_adams_degree()
        return self._to_adams(
            operands, max_adams_degree=max_adams_degree, as_symbol=as_symbol
        )

    def to_lambda(self, as_symbol: bool = False, *, optimize: bool = True) -> sp.Expr:
        """
        Converts this expression into an equivalent lambda polynomial, a polynomial
        depending only on lambda operations of the leaves of the given expression tree.

        Args:
        -----
        as_symbol : bool, optional
            Whether to represent the Adams operators as symbols. Defaults to False.
        optimize : bool, optional
            If True, optimizations are applied during the conversion. Defaults to True.

        Returns:
        --------
        sp.Expr
            A polynomial of Lambda operators equivalent to this expression.
        """
        operands: Set[Operand] = self.free_symbols
        max_adams_degree = self.get_max_adams_degree()

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

        for operand in operands:
            adams_pol = operand._subs_adams(adams_pol, max_adams_degree, as_symbol)

        return adams_pol

    @typechecked
    def sigma(self, degree: int) -> sp.Expr:
        """
        Applies the sigma operation of a given degree to this expression.

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
        Applies the lambda operation of a given degree to this expression.

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

        if degree == 0:
            return sp.Integer(1)
        if degree == 1:
            return self

        return Lambda_(degree, self)

    @typechecked
    def adams(self, degree: int) -> sp.Expr:
        """
        Applies the Adams operation of a given degree to this expression.

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

        if degree == 0:
            return sp.Integer(1)
        if degree == 1:
            return self

        return Adams(degree, self)

    def _to_adams(
        self, operands: Set[Operand], max_adams_degree: int, as_symbol: bool = False
    ) -> sp.Expr:
        """
        Converts this expression subtree into an equivalent Adams polynomial.

        Args:
        -----
        operands : Set[Operand]
            The set of all operands in the expression tree.
        max_adams_degree : int
            The maximum Adams degree in the expression.
        as_symbol : bool, optional
            Whether to represent the Adams operators as symbols. Defaults to False.

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
        self,
        operands: Set[Operand],
        max_adams_degree: int,
        as_symbol: bool = False,
        adams_degree: int = 1,
    ) -> sp.Expr:
        """
        Converts this expression subtree into an equivalent Adams polynomial with optimizations for lambda conversion.
        It converts only lambda operations into Adams polynomials if needed and leaves subexpressions that are
        already lambda polynomials unaffected if no Adams, lambda, or sigma operation acts on them.

        This method is used when the `optimize` flag is set to True in the `to_lambda` method.

        Args:
        -----
        operands : Set[Operand]
            The set of all operands in the expression tree.
        max_adams_degree : int
            The maximum Adams degree in the expression.
        as_symbol : bool, optional
            Whether to represent the Adams operators as symbols. Defaults to False.
        adams_degree : int, optional
            Sum of the degrees of all Adams operators higher than this node in its branch. Defaults to 1.

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
