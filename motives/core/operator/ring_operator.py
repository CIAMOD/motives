import sympy as sp

from ..node import Node
from ..operand import Operand
from ..lambda_context import LambdaContext

from .unary_operator import UnaryOperator
from .nary_operator import NaryOperator

class RingOperator(UnaryOperator):
    """
    An abstract node in an expression tree that represents a ring operator.

    Parameters:
    -----------
    degree : int
        The degree of the ring operator.
    child : Node
        The child node of the ring operator.
    parent : Node
        The parent node of the ring operator.
    """

    def __init__(self, degree: int, child: Node, parent: Node = None):
        super().__init__(child, parent)
        self.degree: int = degree

    def _get_max_adams_degree(self) -> int:
        """
        Returns the maximum adams degree of this subtree once converted to an adams
        polynomial.
        """
        return self.degree * self.child._get_max_adams_degree()

    @property
    def sympy(self) -> sp.Symbol:
        """
        The sympy representation of the ring operator.
        """
        return sp.Function(f"{self}")(self.child.sympy)
    

class Sigma(RingOperator):
    """
    A unary operator node in an expression tree that represents the sigma operation.

    Parameters:
    -----------
    degree : int
        The degree of the sigma operator.
    child : Node
        The child node of the sigma operator.
    parent : Node
        The parent node of the sigma operator.

    Properties:
    -----------
    sympy : sp.Symbol
        The sympy representation of the sigma operator.
    """

    def __init__(self, degree: int, child: Node, parent: Node = None):
        super().__init__(degree, child, parent)

    def __repr__(self) -> str:
        return f"σ{self.degree}"

    def _get_max_groth_degree(self) -> int:
        """
        Returns the degree of the highest degree sigma or lambda operator in the tree.
        """
        return max(self.degree, self.child._get_max_groth_degree())

    def _to_adams(
        self, operands: set[Operand], group_context: LambdaContext
    ) -> sp.Expr:
        """
        Converts this subtree into an equivalent adams polynomial. It calls _to_adams
        on the child, then gets the jth adams of the result (j from 0 to its degree)
        and substitutes it in the adams_to_sigma polynomial.
        """
        if self.degree == 0:
            return 1

        # Get the polynomial that arrives to the sigma operator by calling _to_adams on the child
        ph = self.child._to_adams(operands, group_context)
        # Create a list of the polynomial ph so that ph_list[j] = ψj(ph) for all j
        ph_list = [ph for _ in range(self.degree + 1)]

        # Apply the adams operators to the polynomials in the list
        for j in range(1, self.degree + 1):
            for operand in operands:
                ph_list[j] = operand._to_adams(j, ph_list[j])

        adams_to_sigma = group_context.get_adams_2_sigma_pol(self.degree)

        # Substitute the jth polynomial for the adams of degree j in the adams_to_sigma polynomial
        return adams_to_sigma.xreplace(
            {group_context.adams_vars[i]: ph_list[i] for i in range(self.degree + 1)}
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
        if self.degree == 0:
            return 1

        # Optimization: if the child is an operand and there are no adams operators on top,
        # we can convert the sigma to lambda directly
        if isinstance(self.child, Operand) and adams_degree == 1:
            lambda_to_sigma = group_context.get_lambda_2_sigma_pol(self.degree)
            return lambda_to_sigma.xreplace(
                {
                    group_context.lambda_vars[i]: self.child.get_lambda_var(i)
                    for i in range(self.degree + 1)
                }
            )

        # Get the polynomial that arrives to the sigma operator by calling _to_adams on the child
        ph = self.child._to_adams_lambda(
            operands, group_context, adams_degree + self.degree
        )

        if self.degree == 1:
            return ph

        # Create a list of the polynomial ph so that ph_list[j] = ψj(ph) for all j
        ph_list = [ph for _ in range(self.degree + 1)]

        # Apply the adams operators to the polynomials in the list
        for j in range(1, self.degree + 1):
            for operand in operands:
                ph_list[j] = operand._to_adams(j, ph_list[j])

        adams_to_sigma = group_context.get_adams_2_sigma_pol(self.degree)

        # Substitute the jth polynomial for the adams of degree j in the adams_to_sigma polynomial
        return adams_to_sigma.xreplace(
            {group_context.adams_vars[i]: ph_list[i] for i in range(self.degree + 1)}
        )
    
class Lambda(RingOperator):
    """
    A unary operator node in an expression tree that represents the lambda operation.

    Parameters:
    -----------
    degree : int
        The degree of the lambda operator.
    child : Node
        The child node of the lambda operator.
    parent : Node
        The parent node of the lambda operator.
    """

    def __init__(self, degree: int, child: Node, parent: Node = None):
        super().__init__(degree, child, parent)

    def __repr__(self) -> str:
        return f"λ{self.degree}"

    def _get_max_groth_degree(self) -> int:
        """
        Returns the degree of the highest degree sigma or lambda operator in the tree.
        """
        return max(self.degree, self.child._get_max_groth_degree())

    def _to_adams(
        self, operands: set[Operand], group_context: LambdaContext
    ) -> sp.Expr:
        """
        Converts this subtree into an equivalent adams polynomial. It calls _to_adams
        on the child, then gets the jth adams of the result (j from 0 to its degree)
        and substitutes it in the adams_to_lambda polynomial.
        """
        if self.degree == 0:
            return 1

        # Get the polynomial that arrives to the lambda operator by calling _to_adams on the child
        ph = self.child._to_adams(operands, group_context)
        # Create a list of the polynomial ph so that ph_list[j] = ψj(ph) for all j
        ph_list = [ph for _ in range(self.degree + 1)]

        # Apply the adams operators to the polynomials in the list
        for j in range(1, self.degree + 1):
            for operand in operands:
                ph_list[j] = operand._to_adams(j, ph_list[j])

        adams_to_lambda = group_context.get_adams_2_lambda_pol(self.degree)

        # Substitute the jth polynomial for the adams of degree j in the adams_to_lambda polynomial
        return adams_to_lambda.xreplace(
            {group_context.adams_vars[i]: ph_list[i] for i in range(self.degree + 1)}
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
        if self.degree == 0:
            return 1

        # Optimization: if the child is an operand and there are no adams operators on top,
        # we can return the lambda variable directly
        if isinstance(self.child, Operand) and adams_degree == 1:
            return self.child.get_lambda_var(self.degree)

        # Get the polynomial that arrives to the lambda operator by calling _to_adams on the child
        ph = self.child._to_adams_lambda(
            operands, group_context, adams_degree + self.degree
        )

        if self.degree == 1:
            return ph

        # Create a list of the polynomial ph so that ph_list[j] = ψj(ph) for all j
        ph_list = [ph for _ in range(self.degree + 1)]

        # Apply the adams operators to the polynomials in the list
        for j in range(1, self.degree + 1):
            for operand in operands:
                ph_list[j] = operand._to_adams(j, ph_list[j])

        adams_to_lambda = group_context.get_adams_2_lambda_pol(self.degree)

        # Substitute the jth polynomial for the adams of degree j in the adams_to_lambda polynomial
        return adams_to_lambda.xreplace(
            {group_context.adams_vars[i]: ph_list[i] for i in range(self.degree + 1)}
        )
    
class Adams(RingOperator):
    """
    A unary operator node in an expression tree that represents the lambda operation.

    Parameters:
    -----------
    degree : int
        The degree of the lambda operator.
    child : Node
        The child node of the lambda operator.
    parent : Node
        The parent node of the lambda operator.
    """

    def __init__(self, degree: int, child: Node, parent: Node = None):
        super().__init__(degree, child, parent)

    def __repr__(self) -> str:
        return f"ψ{self.degree}"

    def _get_max_groth_degree(self) -> int:
        """
        Returns the degree of the highest degree sigma or lambda operator in the tree.
        """
        return self.child._get_max_groth_degree()

    def to_adams_local(self):
        if isinstance(self.child, Adams):
            # We add the degrees of the Adams operators and remove the child
            self.degree += self.child.degree
            self.child = self.child.child
            self.child.parent = self

        elif isinstance(self.child, NaryOperator):
            # We move the Adams operator as the children of the NaryOperator,
            # using that it is additive and multiplicative
            for child in self.child.children.copy():
                new_adams = Adams(self.degree, child, self.child)
                child.parent = new_adams
                self.child.children.remove(child)
                self.child.children.add(new_adams)
                new_adams.to_adams()  # We repeat the process for the new Adams operator

            # We remove the Adams operator from the tree, now that it is moved
            if isinstance(self.parent, UnaryOperator):
                self.parent.child = self.child
            elif isinstance(self.parent, NaryOperator):
                self.parent.children.remove(self)
                self.parent.children.add(self.child)

            self.child.parent = self.parent

    def _to_adams(
        self, operands: set[Operand], group_context: LambdaContext
    ) -> sp.Expr:
        """
        Converts this subtree into an equivalent adams polynomial. Calls _to_adams on the
        child, then returns the `self.degree` adams of the result.
        """
        if self.degree == 0:
            return 1

        # Get the polynomial that arrives to the Adams operator by calling _to_adams on the child
        ph = self.child._to_adams(operands, group_context)

        if self.degree == 1:
            return ph

        # Add the degree of the adams operator to all the operands
        for operand in operands:
            ph = operand._to_adams(self.degree, ph)

        return ph

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
        if self.degree == 0:
            return 1

        # Get the polynomial that arrives to the Adams operator by calling _to_adams on the child
        ph = self.child._to_adams_lambda(
            operands, group_context, adams_degree + self.degree
        )

        if self.degree == 1:
            return ph

        # Add the degree of the adams operator to all the operands
        for operand in operands:
            ph = operand._to_adams(self.degree, ph)

        return ph