import sympy as sp
from collections import Counter
from typing import Optional

from ..node import Node
from ..integer import Integer
from ..operand import Operand
from ..lambda_context import LambdaContext

from .unary_operator import Power


class NaryOperator(Node):
    """
    An abstract n-ary operator node in an expression tree.

    Parameters:
    -----------
    children : list[Node]
        The children nodes of the n-ary operator.
    parent : Node
        The parent node of the n-ary operator.
    """

    def __init__(self, children: list[Node], parent: Optional[Node] = None):
        super().__init__(parent)
        self.children: list[Node] = children

    def _get_max_adams_degree(self) -> int:
        """
        Returns the maximum adams degree of this subtree once converted to an adams
        polynomial.
        """
        return max(child._get_max_adams_degree() for child in self.children)

    def _get_max_groth_degree(self) -> int:
        """
        Returns the degree of the highest degree sigma or lambda operator in the tree.
        """
        return max(child._get_max_groth_degree() for child in self.children)

class Multiply(NaryOperator):
    """
    A n-ary operator node in an expression tree that represents multiplication.

    Parameters:
    -----------
    children : list[Node]
        The children nodes of the multiply operator.
    parent : Node
        The parent node of the multiply operator.

    Properties:
    -----------
    sympy : sp.Symbol
        The sympy representation of the multiply operator.
    """

    def __init__(self, children: list[Node], parent: Node = None):
        children = Multiply._flatten(children)
        children = Multiply._join_integers(children)
        children = Multiply._collect(children)
        super().__init__(children, parent)

    @staticmethod
    def _join_integers(children: list) -> list[Node]:
        """
        Joins integers in a list of children.
        Multiply(2, 3, 4, 5) -> Multiply(120).

        Parameters
        ----------
        children : list[Node]
            The children of the multiply operator.

        Returns
        -------
        The list of children with consecutive integers joined.
        """
        new_children = []
        current = 1
        for child in children:
            if isinstance(child, Integer):
                current *= child.value
            elif isinstance(child, int):
                current *= child
            else:
                from ..expression_tree import ET  # Local import to avoid circular dependency
                if isinstance(child, ET):
                    new_children.append(child.root)
                elif isinstance(child, Node):
                    new_children.append(child)
                else:
                    raise TypeError(
                        f"Expected an integer, ET or Node, got {type(child)}."
                    )

        if current != 1:
            new_children = [Integer(current)] + new_children

        return new_children

    @staticmethod
    def _flatten(children: list) -> list:
        """
        Converts Multiply(x, y, Multiply(z, w)) into Multiply(x, y, z, w).

        Parameters
        ----------
        children : list[Node]
            The children of the multiply operator.

        Returns
        -------
        The flattened list of children.
        """
        new_children = []
        for child in children:
            if isinstance(child, Multiply):
                new_children += child.children
            else:
                new_children.append(child)
        return new_children

    @staticmethod
    def _collect(children: list[Node]) -> list[Node]:
        """
        Collects like terms in the children of the multiply operator.
        Multiply(x, x, y) -> Multiply(x^2, y).

        Parameters
        ----------
        children : list[Node]
            The children of the multiply operator.

        Returns
        -------
        The collected list of children.
        """
        collected = Counter()
        new_children = []
        for child in children:
            if isinstance(child, Power):
                if isinstance(child.child, Operand):
                    collected[child.child] += child.degree
                else:
                    new_children.append(child)
            else:
                if isinstance(child, Operand) and not isinstance(child, Integer):
                    collected[child] += 1
                else:
                    new_children.append(child)

        return new_children + [
            (Power(degree, child, parent=children[0].parent) if degree != 1 else child)
            for child, degree in collected.items()
        ]

    def __repr__(self) -> str:
        return "*"

    def _to_adams(
        self, operands: set[Operand], group_context: LambdaContext
    ) -> sp.Expr:
        """
        Converts this subtree into an equivalent adams polynomial. It calls _to_adams
        on each child and multiplies the results.
        """
        return sp.Mul(
            *(child._to_adams(operands, group_context) for child in self.children)
        )

    def _to_adams_lambda(
        self,
        operands: set[Operand],
        group_context: LambdaContext,
        adams_degree: int = 1,
    ) -> sp.Expr:
        """
        Converts this subtree into an equivalent adams polynomial, when applying
        to_lambda. Keeps some lambda variables as they are.

        Parameters
        ----------
        operands : set[Operand]
            The operands in the tree.
        group_context : LambdaContext
            The group context of the tree.
        adams_degree : int
            The degree of the adams operators on top of this node.

        Returns
        -------
        The equivalent adams polynomial with some lambda variables.
        """
        return sp.Mul(
            *(
                child._to_adams_lambda(operands, group_context, adams_degree)
                for child in self.children
            )
        )

    @property
    def sympy(self) -> sp.Symbol:
        """
        Returns the sympy representation of the multiply operator.
        """
        return sp.Mul(*(child.sympy for child in self.children))

class Add(NaryOperator):
    """
    A n-ary operator node in an expression tree that represents addition.

    Parameters:
    -----------
    children : list[Node]
        The children nodes of the add operator.
    parent : Node
        The parent node of the add operator.

    Properties:
    -----------
    sympy : sp.Symbol
        The sympy representation of the add operator.
    """

    def __init__(self, children: list[Node], parent: Node = None):
        children = Add._flatten(children)
        children = Add._join_integers(children)
        children = Add._collect(children)
        super().__init__(children, parent)

    @staticmethod
    def _join_integers(children: list) -> list[Node]:
        """
        Joins integers in a list of children.
        Add(1, 2, 3, 4, 5) -> Add(15).

        Parameters
        ----------
        children : list[Node]
            The children of the add operator.

        Returns
        -------
        The list of children with consecutive integers joined.
        """
        new_children = []
        current = 0
        for child in children:
            if isinstance(child, Integer):
                current += child.value
            elif isinstance(child, int):
                current += child
            else:
                from ..expression_tree import ET  # Local import to avoid circular dependency
                if isinstance(child, ET):
                    new_children.append(child.root)
                elif isinstance(child, Node):
                    new_children.append(child)
                else:
                    raise TypeError(
                        f"Expected an integer, ET or Node, got {type(child)}."
                    )

        if current != 0:
            new_children = [Integer(current)] + new_children

        return new_children

    @staticmethod
    def _flatten(children: list) -> list:
        """
        Converts Add(x, y, Add(z, w)) into Add(x, y, z, w).

        Parameters
        ----------
        children : list[Node]
            The children of the add operator.

        Returns
        -------
        The flattened list of children.
        """
        new_children = []
        for child in children:
            if isinstance(child, Add):
                new_children += child.children
            else:
                new_children.append(child)
        return new_children

    @staticmethod
    def _collect(children: list[Node]) -> list[Node]:
        """
        Collects like terms in the children of the add operator.
        Add(x, x, y) -> Add(2x, y). Add(2*x, x, y) -> Add(3*x, y).

        Parameters
        ----------
        children : list[Node]
            The children of the add operator.

        Returns
        -------
        The collected list of children.
        """
        collected = Counter()
        new_children = []
        for child in children:
            if (
                isinstance(child, Multiply)
                and len(child.children) == 2
                and isinstance(child.children[0], Integer)
                and isinstance(child.children[1], Operand)
            ):
                count = child.children[0].value
                collected[child.children[1]] += count
            else:
                if isinstance(child, Operand) and not isinstance(child, Integer):
                    collected[child] += 1
                else:
                    new_children.append(child)

        return new_children + [
            (
                Multiply([Integer(count), child], parent=children[0].parent)
                if count != 1
                else child
            )
            for child, count in collected.items()
        ]

    def __repr__(self) -> str:
        return f"+"

    def _to_adams(
        self, operands: set[Operand], group_context: LambdaContext
    ) -> sp.Expr:
        """
        Converts this subtree into an equivalent adams polynomial. It calls _to_adams
        on each child and sums the results.
        """
        return sp.Add(
            *(child._to_adams(operands, group_context) for child in self.children)
        )

    def _to_adams_lambda(
        self,
        operands: set[Operand],
        group_context: LambdaContext,
        adams_degree: int = 1,
    ) -> sp.Expr:
        """
        Converts this subtree into an equivalent adams polynomial, when applying
        to_lambda. Keeps some lambda variables as they are.

        Parameters
        ----------
        operands : set[Operand]
            The operands in the tree.
        group_context : LambdaContext
            The group context of the tree.
        adams_degree : int
            The degree of the adams operators on top of this node.

        Returns
        -------
        The equivalent adams polynomial with some lambda variables.
        """
        return sp.Add(
            *(
                child._to_adams_lambda(operands, group_context, adams_degree)
                for child in self.children
            )
        )

    @property
    def sympy(self) -> sp.Symbol:
        """
        The sympy representation of the add operator.
        """
        return sp.Add(*(child.sympy for child in self.children))