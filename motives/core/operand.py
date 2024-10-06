# operand.py

from __future__ import annotations  # For forward references
from typing import Optional, Hashable, Set, TypeVar
import sympy as sp
import hashlib
from multipledispatch import dispatch

from .node import Node
from .lambda_context import LambdaContext

ET = TypeVar('ET')  # Define ET as a TypeVar for type hinting

class Operand(Node):
    """
    An abstract node in an expression tree that represents an operand.

    Parameters:
    -----------
    parent : Node
        The parent node of the operand. If the operand is the root
        of the tree, parent is None.
    id_ : Optional[hashable]
        The id of the operand, used to identify it (if two operands
        have the same id, they are the same operand).

    Methods:
    --------
    sigma(degree: int) -> ET
        Applies the sigma operation to the current operand.
    lambda_(degree: int) -> ET
        Applies the lambda operation to the current operand.
    adams(degree: int) -> ET
        Applies the adams operation to the current operand.

    Properties:
    -----------
    sympy : sp.Expr
        The sympy representation of the node (and any children).
    """

    def __init__(self, parent: Optional[Node] = None, id_: Optional[Hashable] = None):
        if id_ is None:
            self.id = self._generate_id()
        else:
            self.id = id_
        super().__init__(parent)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Operand):
            return False
        return self.id == other.id

    def _generate_id(self) -> str:
        """
        Generates the id of the operand from its attributes.
        """
        block_string = str(tuple(sorted(self.__dict__.items()))) + str(self.__class__)
        return hashlib.sha256(block_string.encode()).hexdigest()

    # All the overloads are sent to the correct ET overloads
    def _add(self, other: object) -> ET:
        from .expression_tree import ET  # Local import to avoid circular dependency
        return ET(self) + other

    def __add__(self, other: object) -> ET:
        return self._add(other)

    def __radd__(self, other: object) -> ET:
        return self._add(other)

    def __sub__(self, other: object) -> ET:
        from .expression_tree import ET  # Local import to avoid circular dependency
        return ET(self) - other

    def __rsub__(self, other: object) -> ET:
        from .expression_tree import ET  # Local import to avoid circular dependency
        return -ET(self) + other

    def __neg__(self) -> ET:
        from .expression_tree import ET  # Local import to avoid circular dependency
        return -ET(self)

    def _mul(self, other: object) -> ET:
        from .expression_tree import ET  # Local import to avoid circular dependency
        return ET(self) * other

    def __mul__(self, other: object) -> ET:
        return self._mul(other)

    def __rmul__(self, other: object) -> ET:
        return self._mul(other)

    def __truediv__(self, other: object) -> ET:
        from .expression_tree import ET  # Local import to avoid circular dependency
        return ET(self) / other

    def __rtruediv__(self, other: object) -> ET:
        from .expression_tree import ET  # Local import to avoid circular dependency
        return other / ET(self)

    def __pow__(self, other: int) -> ET:
        from .expression_tree import ET  # Local import to avoid circular dependency
        return ET(self) ** other

    def sigma(self, degree: int) -> ET:
        """
        Applies the sigma operation to the current operand.

        Parameters
        ----------
        degree : int
            The degree of the sigma operator.

        Returns
        -------
        An expression tree with the sigma operator applied.
        """
        from .expression_tree import ET  # Local import to avoid circular dependency
        return ET(self).sigma(degree)

    def lambda_(self, degree: int) -> ET:
        """
        Applies the lambda operation to the current operand.

        Parameters
        ----------
        degree : int
            The degree of the lambda operator.

        Returns
        -------
        An expression tree with the lambda operator applied.
        """
        from .expression_tree import ET  # Local import to avoid circular dependency
        return ET(self).lambda_(degree)

    def adams(self, degree: int) -> ET:
        """
        Applies the adams operation to the current operand.

        Parameters
        ----------
        degree : int
            The degree of the adams operator.

        Returns
        -------
        An expression tree with the adams operator applied.
        """
        from .expression_tree import ET  # Local import to avoid circular dependency
        return ET(self).adams(degree)

    def _get_max_adams_degree(self) -> int:
        """
        The maximum adams degree of this subtree once converted to an adams polynomial.
        """
        return 1

    def _get_max_groth_degree(self) -> int:
        """
        Returns the degree of the highest degree sigma or lambda operator in the tree.
        """
        return 0

    def get_adams_var(self, i: int) -> sp.Symbol | int:
        """
        Returns the adams variable of degree i.

        Parameters
        ----------
        i : int
            The degree of the adams variable.

        Returns
        -------
        The adams variable of degree i.
        """
        raise NotImplementedError("This method must be implemented in all subclasses.")

    def get_lambda_var(self, i: int) -> sp.Symbol | int:
        """
        Returns the lambda variable of degree i.

        Parameters
        ----------
        i : int
            The degree of the lambda variable.

        Returns
        -------
        The lambda variable of degree i.
        """
        raise NotImplementedError("This method must be implemented in all subclasses.")

    @dispatch(int, sp.Expr)
    def _to_adams(self, degree: int, ph: sp.Expr) -> sp.Expr:
        """
        Applies an adams operation of degree `degree` to any instances of this operand
        in the polynomial `ph`.
        """
        raise NotImplementedError("This method must be implemented in all subclasses.")

    @dispatch(set, LambdaContext)
    def _to_adams(
        self, operands: set[Operand], group_context: LambdaContext
    ) -> sp.Expr:
        """
        Converts this subtree into an equivalent adams polynomial.
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
        """
        return self._to_adams(operands, group_context)

    def _subs_adams(self, group_context: LambdaContext, ph: sp.Expr) -> sp.Expr:
        """
        Substitutes any instances of an adams of this operand (ψ_d(operand)) into its
        equivalent polynomial of lambdas.
        """
        raise NotImplementedError("This method must be implemented in all subclasses.")

    @property
    def sympy(self) -> sp.Symbol:
        """
        The sympy representation of the operand.
        """
        raise NotImplementedError(
            "This property must be implemented in all subclasses."
        )
