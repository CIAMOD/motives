# node.py

from __future__ import annotations  # For forward references in type hints
from typing import Optional, Set, TypeVar
import sympy as sp

from .lambda_context import LambdaContext

Operand = TypeVar('Operand')  # Define Operand as a TypeVar for type hinting
ET = TypeVar('ET')  # Define Operand as a TypeVar for type hinting


class Node:
    """
    An abstract node in an expression tree.

    Parameters:
    -----------
    parent : Node
        The parent node of the current node. If the
        node is the root of the tree, parent is None.

    Properties:
    -----------
    sympy : sp.Expr
        The sympy representation of the node (and any children)
    """

    def __init__(self, parent: Optional[Node] = None) -> None:
        self.parent: Node = parent

    @property
    def sympy(self) -> sp.Expr:
        """
        The sympy representation of the node.
        """
        raise NotImplementedError(
            "This property must be implemented in all subclasses."
        )

    def __eq__(self, other: Node | ET) -> bool:
        if not isinstance(other, (Node, ET)):
            return False
        return self.sympy - other.sympy == 0

    def __hash__(self) -> int:
        return hash(self.sympy)

    def _get_max_adams_degree(self) -> int:
        """
        The maximum adams degree of this subtree once converted to an adams polynomial.
        """
        raise NotImplementedError("This method must be implemented in all subclasses.")

    def _get_max_groth_degree(self) -> int:
        """
        Returns the degree of the highest degree sigma or lambda operator in the tree.
        """
        raise NotImplementedError("This method must be implemented in all subclasses.")

    def _to_adams(
        self, operands: set[Operand], group_context: LambdaContext
    ) -> sp.Expr:
        """
        Converts this subtree into an equivalent adams polynomial.

        Parameters
        ----------
        operands : set[Operand]
            The set of all operands in the expression tree.
        group_context : LambdaContext
            The context of the Grothendieck group.

        Returns
        -------
        The equivalent adams polynomial.
        """
        raise NotImplementedError("This method must be implemented in all subclasses.")

    def _to_adams_lambda(
        self,
        operands: set[Operand],
        group_context: LambdaContext,
        adams_degree: int = 1,
    ) -> sp.Expr:
        """
        Converts this subtree into an equivalent adams polynomial.

        Parameters
        ----------
        operands : set[Operand]
            The set of all operands in the expression tree.
        group_context : LambdaContext
            The context of the Grothendieck group.
        adams_degree : int
            The degree of the adams operator.

        Returns
        -------
        The equivalent adams polynomial.
        """
        raise NotImplementedError("This method must be implemented in all subclasses.")