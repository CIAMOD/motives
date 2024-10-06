from __future__ import annotations

import sympy as sp
from multipledispatch import dispatch
from typing import Optional
from typeguard import typechecked

from ..grothendieck_motives.curves.curve import Curve
from ..grothendieck_motives.curves.hodge import Hodge
from ..grothendieck_motives.point import Point
from ..grothendieck_motives.lefschetz import Lefschetz
from ..grothendieck_motives.symbol import Symbol

from .operator.nary_operator import NaryOperator, Add, Multiply
from .operator.unary_operator import UnaryOperator, Subtract, Divide, Power
from .operator.ring_operator import RingOperator, Sigma, Lambda, Adams

from .lambda_context import LambdaContext
from .node import Node
from .operand import Operand
from .variable import Variable
from .integer import Integer

class ET:
    """
    A class to represent as a tree a mathematical expression in a lambda ring
    with sigma and adams operations.

    Attributes
    ----------
    root : Node
        The root node of the tree
    operands : dict[Operand, list]
        A dictionary that maps the operands of the tree to their adams variables.

    Methods
    -------
    sigma(degree: int) -> ET
        Applies the sigma operation to the current tree.
    lambda_(degree: int) -> ET
        Applies the lambda operation to the current tree.
    adams(degree: int) -> ET
        Applies the adams operation to the current tree.
    to_adams() -> sp.Expr
        Returns the polynomial of adams operators equivalent to the current tree.
    get_max_adams_degree() -> int
        Returns the max degree of the adams operator in the tree.
    get_max_groth_degree() -> int
        Returns the degree of the highest degree sigma or lambda operator in the tree.
    to_lambda() -> sp.Expr
        Returns the polynomial of lambda operators equivalent to the current tree.
    subs(subs_dict: dict[Operand, Operand | ET]) -> ET
        Substitutes the operands specified in the dictionary by the corresponding
        operand or expression tree.
    clone() -> ET
        Returns a copy of the current tree.

    Class Methods
    -------------
    from_sympy(expr: sp.Expr) -> ET
        Returns an expression tree equivalent to the sympy expression.

    Properties
    ----------
    sympy : sp.Expr
        The sympy representation of the tree.
    """

    def __init__(
        self, root: Optional[Node] = None, operands: Optional[set[Operand]] = None
    ):
        self.root = root

        if operands is None:
            # Get the operands from the tree
            self.operands: set[Operand] = set()
            self._get_operands(self.root)
            # Add the lefschetz operator to the operands if there are curves in the tree
        else:
            self.operands: set[Operand] = operands

        for operand in [op for op in self.operands]:
            if isinstance(operand, Curve):
                # Discard the curve_hodge from the operands because the curve is already applying
                # the necessary transformations, and if there were both it would be done twice
                self.operands.discard(operand.curve_hodge)

                # Add a point and a lefschetz operator to the operands, because the curve is not
                # applying their transformations. This is done to avoid applying them more than
                # once in the case that there are multiple curves in the tree
                self.operands.add(Point())
                self.operands.add(Lefschetz())

    def __repr__(self) -> str:
        return self._repr(self.root)

    def _repr(self, node: Node) -> str:
        if isinstance(node, Operand):
            return str(node)
        elif isinstance(node, Divide):
            return f"{self._repr(node.child)}^(-1)"
        elif isinstance(node, Power):
            return f"{self._repr(node.child)}^{node.degree}"
        elif isinstance(node, RingOperator):
            return f"{node}({self._repr(node.child)})"
        elif isinstance(node, UnaryOperator):
            return f"{node}{self._repr(node.child)}"
        elif isinstance(node, NaryOperator):
            return (
                "("
                + f" {node} ".join((self._repr(child) for child in node.children))
                + ")"
            )

    def __eq__(self, other: ET) -> bool:
        if not isinstance(other, ET):
            return NotImplemented
        return (self.sympy - other.sympy).simplify() == 0

    @dispatch(Operand)
    def _add(self, other: Operand) -> ET:
        if isinstance(self.root, Add):
            # If self.root is an add, add other to the children
            new_root = Add([*self.root.children, other])
            other.parent = new_root
            for child in self.root.children:
                child.parent = new_root

        else:
            # If self.root is not an add, create a new add with self.root and other as children
            new_root = Add([self.root, other])
            self.root.parent = other.parent = new_root

        return ET(new_root, self.operands | {other})

    @dispatch(int)
    def _add(self, other: int) -> ET:
        if other == 0:
            return self
        else:
            return self + ET(Integer(value=other))

    @dispatch(object)
    def _add(self, other: ET) -> ET:
        if not isinstance(other, ET):
            if isinstance(other, Node):
                other = ET(other)
            else:
                # Only options for adding are integers, Nodes and other ETs
                return NotImplemented

        if isinstance(self.root, Add):
            if isinstance(other.root, Add):
                # If both roots are adds, add the children
                new_root = Add([*self.root.children, *other.root.children])
                for child in other.root.children:
                    child.parent = new_root
            else:
                # If only the first is an add, add other to the children
                new_root = Add([*self.root.children, other.root])
                other.root.parent = self.root

            for child in self.root.children:
                child.parent = new_root

        elif isinstance(other.root, Add):
            # If only the second is an add, add self.root to the add's children
            new_root = Add([self.root, *other.root.children])
            self.root.parent = other.root
            for child in other.root.children:
                child.parent = new_root

        else:
            # If none are adds, create a new add with both as children
            new_root = Add([self.root, other.root])
            self.root.parent = other.root.parent = new_root

        return ET(new_root, self.operands | other.operands)

    def __add__(self, other: object) -> ET:
        return self._add(other)

    def __radd__(self, other: object) -> ET:
        return self._add(other)

    @dispatch(Operand)
    def __sub__(self, other: Operand) -> ET:
        new_root = Subtract(other)
        other.parent = new_root
        return self + ET(new_root)

    @dispatch(int)
    def __sub__(self, other: int) -> ET:
        if other == 0:
            return self
        else:
            return self + ET(Integer(-other))

    @dispatch(object)
    def __sub__(self, other: ET) -> ET:
        if not isinstance(other, ET):
            if isinstance(other, Node):
                other = ET(other)
            else:
                # Only options for subtracting are integers, Nodes and other ETs
                return NotImplemented
        new_root = Subtract(other.root)
        other.root.parent = new_root
        return self + ET(new_root)

    def __rsub__(self, other: object) -> ET:
        return -self + other

    def __neg__(self) -> ET:
        if isinstance(self.root, Subtract):
            # If self.root is a subtract, return the child
            return ET(self.root.child, self.operands.copy())

        new_root = Subtract(self.root)
        self.root.parent = new_root
        return ET(new_root, self.operands.copy())

    @dispatch(Operand)
    def _mul(self, other: Operand) -> ET:
        if isinstance(self.root, Multiply):
            # If self.root is a multiply, add the other to the children
            new_root = Multiply([*self.root.children, other])
            for child in self.root.children:
                child.parent = new_root
            other.parent = new_root

        else:
            # If self.root is not a multiply, create a new multiply with self.root and other as children
            new_root = Multiply([self.root, other])
            self.root.parent = other.parent = new_root
        return ET(new_root, self.operands | {other})

    @dispatch(int)
    def _mul(self, other: int) -> ET | int:
        if other == 1:
            return self
        elif other == 0:
            return 0
        else:
            return self * ET(Integer(value=other))

    @dispatch(object)
    def _mul(self, other: ET) -> ET:
        if not isinstance(other, ET):
            if isinstance(other, Node):
                other = ET(other)
            else:
                # Only options for multiplying are integers, Nodes and other ETs
                return NotImplemented

        if isinstance(self.root, Multiply):
            if isinstance(other.root, Multiply):
                # If both roots are multiplies, add the children
                new_root = Multiply([*self.root.children, *other.root.children])
                for child in other.root.children:
                    child.parent = self.root
            else:
                # If only the first is a multiply, add the other to the children
                new_root = Multiply([*self.root.children, other.root])
                other.root.parent = self.root

            for child in self.root.children:
                child.parent = self.root

        elif isinstance(other.root, Multiply):
            # If only the second is a multiply, add self.root to the multiply's children
            new_root = Multiply([self.root, *other.root.children])
            self.root.parent = other.root
            for child in other.root.children:
                child.parent = new_root

        else:
            # If none are multiplies, create a new multiply with both as children
            new_root = Multiply([self.root, other.root])
            self.root.parent = other.root.parent = new_root

        return ET(new_root, self.operands | other.operands)

    def __mul__(self, other: object) -> ET:
        return self._mul(other)

    def __rmul__(self, other: object) -> ET:
        return self._mul(other)

    @dispatch(Operand)
    def __truediv__(self, other: Operand) -> ET:
        new_root = Divide(other)
        other.parent = new_root
        return self * ET(new_root)

    @dispatch(int)
    def __truediv__(self, other: int) -> ET:
        if other == 1:
            return self
        elif other == 0:
            raise ZeroDivisionError("Division by zero")
        else:
            return self / ET(Integer(value=other))

    @dispatch(object)
    def __truediv__(self, other: ET) -> ET:
        if not isinstance(other, ET):
            if isinstance(other, Node):
                other = ET(other)
            else:
                # Only options for dividing are integers, Nodes and other ETs
                return NotImplemented

        if isinstance(other.root, Divide):
            # If other.root is a divide, return the child
            return self * ET(other.root.child, other.operands.copy())

        # Otherwise, create a new divide with other as the child
        # and multiply it by self
        new_root = Divide(other.root)
        other.root.parent = new_root
        return self * ET(new_root)

    def __rtruediv__(self, other: object) -> ET:
        if isinstance(self.root, Divide):
            # If self.root is a divide, return the child
            return other * ET(self.root.child, self.operands.copy())

        new_root = Divide(self.root)
        self.root.parent = new_root
        return ET(new_root) * other

    def __pow__(self, other: int) -> ET | int:
        if not isinstance(other, int):
            # Only options for exponents are integers
            return NotImplemented

        if other == 1:
            return self

        if other == -1:
            new_root = Divide(self.root)
            self.root.parent = new_root
            return ET(new_root, self.operands.copy())

        if other == 0:
            return 1

        return ET(Power(other, self.root), self.operands.copy())

    @typechecked
    def sigma(self, degree: int) -> ET:
        """
        Applies the sigma operation to the current tree.

        Parameters
        ----------
        degree : int
            The degree of the sigma operator.

        Returns
        -------
        The current tree with the sigma operator applied.
        """
        new_root = Sigma(degree, self.root)
        self.root.parent = new_root
        return ET(new_root, self.operands.copy())

    @typechecked
    def lambda_(self, degree: int) -> ET:
        """
        Applies the lambda operation to the current tree.

        Parameters
        ----------
        degree : int
            The degree of the lambda operator.

        Returns
        -------
        The current tree with the lambda operator applied.
        """
        new_root = Lambda(degree, self.root)
        self.root.parent = new_root
        return ET(new_root, self.operands.copy())

    @typechecked
    def adams(self, degree: int) -> ET:
        """
        Applies the adams operation to the current tree.

        Parameters
        ----------
        degree : int
            The degree of the adams operator.

        Returns
        -------
        The current tree with the adams operator applied.
        """
        new_root = Adams(degree, self.root)
        self.root.parent = new_root
        return ET(new_root, self.operands.copy())

    def _get_operands(self, node: Node) -> None:
        """
        Returns the operands of the tree.
        """
        if isinstance(node, Operand):
            # If the node is a variable, add it to the operands dictionary and finish recursing
            self.operands.add(node)
        elif isinstance(node, UnaryOperator):
            # If the node is a unary operator, recurse on its child
            self._get_operands(node.child)
        elif isinstance(node, NaryOperator):
            # If the node is an n-ary operator, recurse on its children
            for child in node.children:
                self._get_operands(child)

    def get_max_adams_degree(self) -> int:
        """
        Returns the maximum adams degree of this subtree once converted to an adams
        polynomial.

        Returns
        -------
        The maximum adams degree of the tree.
        """
        return self.root._get_max_adams_degree()

    def get_max_groth_degree(self) -> int:
        """
        Returns the degree of the highest degree sigma or lambda operator in the tree.

        Returns
        -------
        The degree of the highest degree sigma or lambda operator in the tree.
        """
        return self.root._get_max_groth_degree()

    def to_adams(self, group_context: LambdaContext = None) -> sp.Expr:
        """
        Turns the current tree into a polynomial of adams operators.

        Parameters
        ----------
        group_context : LambdaContext
            The group context of the tree. If None, a new one is created.

        Returns
        -------
        The polynomial of adams operators equivalent to the current tree.
        """
        # Initialize a LambdaContext
        if group_context is None:
            group_context = LambdaContext()

        return self.root._to_adams(self.operands, group_context)

    def to_lambda(
        self, group_context: LambdaContext = None, *, optimize=True
    ) -> sp.Expr:
        """
        Turns the current tree into a polynomial of lambda operators.

        Parameters
        ----------
        group_context : LambdaContext
            The group context of the tree. If None, a new one is created.
        optimize : bool
            If True, optimizations are implemented when converting to lambda.

        Returns
        -------
        The polynomial of lambda operators equivalent to the current tree.
        """
        # Initialize a LambdaContext
        if group_context is None:
            group_context = LambdaContext()

        # Get the adams polynomial of the tree
        if optimize:
            adams_pol = self.root._to_adams_lambda(self.operands, group_context, 1)
        else:
            adams_pol = self.root._to_adams(self.operands, group_context)

        if isinstance(adams_pol, (sp.Integer, int)):
            return adams_pol

        # Substitute the adams variables for the lambda variables
        for operand in self.operands:
            adams_pol = operand._subs_adams(group_context, adams_pol)

        return adams_pol

    @typechecked
    def subs(self, subs_dict: dict[Operand, Operand | ET]) -> ET:
        """
        Substitutes the operands in the tree for the values in the dictionary.

        Parameters
        ----------
        subs_dict : dict
            The dictionary with the substitutions to make.

        Returns
        -------
        The (cloned) tree with the substitutions made.
        """
        return self._subs(self.root, subs_dict)

    def _subs(self, node: Node, subs_dict: dict[Operand, Operand | ET]) -> Node:
        if isinstance(node, Operand):
            if node in subs_dict:
                return (
                    subs_dict[node]
                    if isinstance(subs_dict[node], Operand)
                    else subs_dict[node].root
                )
        elif isinstance(node, (RingOperator, Power)):
            new_node = type(node)(node.degree, self._subs(node.child))
            new_node.child.parent = new_node
        elif isinstance(node, UnaryOperator):
            new_node = type(node)(self._subs(node.child))
            new_node.child.parent = new_node
        elif isinstance(node, NaryOperator):
            new_node = type(node)([self._subs(child) for child in node.children])
            for child in new_node.children:
                child.parent = new_node
        return new_node

    @property
    def sympy(self) -> sp.Expr:
        """
        Returns the sympy expression of the tree. Ids aren't kept.
        """
        return self.root.sympy

    @staticmethod
    @typechecked
    def from_sympy(expr: sp.Expr) -> ET:
        """
        Returns an expression tree from a sympy expression.

        Parameters
        ----------
        expr : sp.Expr
            The sympy expression to convert.

        Returns
        -------
        The expression tree equivalent to the sympy expression.
        """
        if isinstance(expr, sp.Symbol):
            if expr.name[:2] == "C_":
                elements = expr.name.split("_")
                return ET(Curve(sp.Symbol(elements[1]), int(elements[2])))
            elif expr.name[:2] == "s_":
                if expr.name[2] == "L":
                    return ET(Lefschetz())
                return ET(Symbol(sp.Symbol(expr.name[2:])))
            elif expr.name[:2] == "h_":
                return ET(Hodge(sp.Symbol(expr.name[2:])))
            elif expr.name == "pt":
                return ET(Point())
            return ET(Variable(expr))

        elif isinstance(expr, sp.Integer):
            return ET(Integer(expr.p))

        elif isinstance(expr, sp.Add):
            return ET(Add([ET.from_sympy(arg) for arg in expr.args]))

        elif isinstance(expr, sp.Mul):
            return ET(Multiply([ET.from_sympy(arg) for arg in expr.args]))

        elif isinstance(expr, sp.Pow):
            return ET.from_sympy(expr.base) ** int(expr.exp)

        elif isinstance(expr, sp.Function):
            if "σ" in expr.func.__name__:
                return ET.from_sympy(expr.args[0]).sigma(int(expr.func.__name__[1:]))
            elif "λ" in expr.func.__name__:
                return ET.from_sympy(expr.args[0]).lambda_(int(expr.func.__name__[1:]))
            elif "ψ" in expr.func.__name__:
                return ET.from_sympy(expr.args[0]).adams(int(expr.func.__name__[1:]))
        else:
            raise ValueError(f"Cannot convert {expr} to an expression tree.")

    def clone(self) -> ET:
        """
        Returns a copy of the current tree.

        Returns
        -------
        A copy of the current tree.
        """
        return ET(self._clone(self.root), self.operands.copy())

    def _clone(self, node: Node) -> Node:
        """
        Returns a copy of the provided node.
        """
        if isinstance(node, Operand):
            return node
        elif isinstance(node, (RingOperator, Power)):
            new_node = type(node)(node.degree, self._clone(node.child))
            new_node.child.parent = new_node
        elif isinstance(node, UnaryOperator):
            new_node = type(node)(self._clone(node.child))
            new_node.child.parent = new_node
        elif isinstance(node, NaryOperator):
            new_node = type(node)([self._clone(child) for child in node.children])
            for child in new_node.children:
                child.parent = new_node
        return new_node


if __name__ == "__main__":
    pass
