import sympy as sp

from ..node import Node
from ..operand import Operand
from ..lambda_context import LambdaContext

class UnaryOperator(Node):
    """
    An abstract unary operator node in an expression tree.

    Parameters:
    -----------
    child : Node
        The child node of the unary operator.
    parent : Node
        The parent node of the unary operator.
    """

    def __init__(self, child: Node, parent: Node = None):
        super().__init__(parent)
        self.child: Node = child

class Subtract(UnaryOperator):
    """
    A unary operator node in an expression tree that represents subtraction.

    Parameters:
    -----------
    child : Node
        The child node of the subtract operator.
    parent : Node
        The parent node of the subtract operator.

    Properties:
    -----------
    sympy : sp.Symbol
        The sympy representation of the subtract operator.
    """

    def __init__(self, child: Node, parent: Node = None):
        super().__init__(child, parent)

    def __repr__(self) -> str:
        return f"-"

    def _get_max_adams_degree(self) -> int:
        """
        Returns the maximum adams degree of this subtree once converted to an adams
        polynomial.
        """
        return self.child._get_max_adams_degree()

    def _get_max_groth_degree(self) -> int:
        """
        Returns the degree of the highest degree sigma or lambda operator in the tree.
        """
        return self.child._get_max_groth_degree()

    def _to_adams(
        self, operands: set[Operand], group_context: LambdaContext
    ) -> sp.Expr:
        """
        Converts this subtree into an equivalent adams polynomial. It calls _to_adams
        on the child and returns the negative of the result.
        """
        return -self.child._to_adams(operands, group_context)

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
        return -self.child._to_adams_lambda(operands, group_context, adams_degree)

    @property
    def sympy(self) -> sp.Symbol:
        """
        Returns the sympy representation of the subtract operator.
        """
        return -self.child.sympy
    
class Divide(UnaryOperator):
    """
    A unary operator node in an expression tree that represents division.

    Parameters:
    -----------
    child : Node
        The child node of the divide operator.
    parent : Node
        The parent node of the divide operator.

    Properties:
    -----------
    sympy : sp.Symbol
        The sympy representation of the divide operator.
    """

    def __init__(self, child: Node, parent: Node = None):
        super().__init__(child, parent)

    def _get_max_adams_degree(self) -> int:
        """
        Returns the maximum adams degree of this subtree once converted to an adams
        polynomial.
        """
        return self.child._get_max_adams_degree()

    def _get_max_groth_degree(self) -> int:
        """
        Returns the degree of the highest degree sigma or lambda operator in the tree.
        """
        return self.child._get_max_groth_degree()

    def _to_adams(
        self, operands: set[Operand], group_context: LambdaContext
    ) -> sp.Expr:
        """
        Converts this subtree into an equivalent adams polynomial. It calls _to_adams
        on the child and returns the inverse of the result.
        """
        return 1 / self.child._to_adams(operands, group_context)

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
        return 1 / self.child._to_adams_lambda(operands, group_context, adams_degree)

    @property
    def sympy(self) -> sp.Symbol:
        """
        Returns the sympy representation of the divide operator.
        """
        return 1 / self.child.sympy
    
class Power(UnaryOperator):
    """
    A unary operator node in an expression tree that represents exponentiation.

    Parameters:
    -----------
    degree : int
        The degree of the power operator.
    child : Node
        The child node of the power operator.
    parent : Node
        The parent node of the power operator.

    Properties:
    -----------
    sympy : sp.Symbol
        The sympy representation of the power operator.
    """

    def __init__(self, degree: int, child: Node, parent: Node = None):
        if not isinstance(degree, int):
            raise TypeError(f"Expected an integer, got {type(degree)}.")
        super().__init__(child, parent)
        self.degree: int = degree

    def _get_max_adams_degree(self) -> int:
        """
        Returns the maximum adams degree of this subtree once converted to an adams
        polynomial.
        """
        return self.child._get_max_adams_degree()

    def _get_max_groth_degree(self) -> int:
        """
        Returns the degree of the highest degree sigma or lambda operator in the tree.
        """
        return self.child._get_max_groth_degree()

    def _to_adams(
        self, operands: set[Operand], group_context: LambdaContext
    ) -> sp.Expr:
        """
        Converts this subtree into an equivalent adams polynomial. It calls _to_adams
        on the child and returns the result raised to the power of the degree.
        """
        return self.child._to_adams(operands, group_context) ** self.degree

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
        return (
            self.child._to_adams_lambda(operands, group_context, adams_degree)
            ** self.degree
        )

    @property
    def sympy(self) -> sp.Symbol:
        """
        Returns the sympy representation of the power operator.
        """
        return self.child.sympy**self.degree