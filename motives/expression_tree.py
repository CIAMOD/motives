from __future__ import annotations

from motives.grothGroupContext import GrothGroupContext
from motives.utils import expr_from_pol

import sympy as sp
from sympy.polys.rings import PolyElement
from sympy.polys.rings import PolyRing
from multipledispatch import dispatch
import hashlib
from typing import Hashable, Optional
from typeguard import typechecked
import math
from collections import Counter


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
        self, operands: set[Operand], group_context: GrothGroupContext
    ) -> sp.Expr:
        """
        Converts this subtree into an equivalent adams polynomial.

        Parameters
        ----------
        operands : set[Operand]
            The set of all operands in the expression tree.
        group_context : GrothGroupContext
            The context of the Grothendieck group.

        Returns
        -------
        The equivalent adams polynomial.
        """
        raise NotImplementedError("This method must be implemented in all subclasses.")

    def _to_adams_lambda(
        self,
        operands: set[Operand],
        group_context: GrothGroupContext,
        adams_degree: int = 1,
    ) -> sp.Expr:
        """
        Converts this subtree into an equivalent adams polynomial.

        Parameters
        ----------
        operands : set[Operand]
            The set of all operands in the expression tree.
        group_context : GrothGroupContext
            The context of the Grothendieck group.
        adams_degree : int
            The degree of the adams operator.

        Returns
        -------
        The equivalent adams polynomial.
        """
        raise NotImplementedError("This method must be implemented in all subclasses.")


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
        return ET(self) + other

    def __add__(self, other: object) -> ET:
        return self._add(other)

    def __radd__(self, other: object) -> ET:
        return self._add(other)

    def __sub__(self, other: object) -> ET:
        return ET(self) - other

    def __rsub__(self, other: object) -> ET:
        return -ET(self) + other

    def __neg__(self) -> ET:
        return -ET(self)

    def _mul(self, other: object) -> ET:
        return ET(self) * other

    def __mul__(self, other: object) -> ET:
        return self._mul(other)

    def __rmul__(self, other: object) -> ET:
        return self._mul(other)

    def __truediv__(self, other: object) -> ET:
        return ET(self) / other

    def __rtruediv__(self, other: object) -> ET:
        return other / ET(self)

    def __pow__(self, other: int) -> ET:
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

    @dispatch(set, GrothGroupContext)
    def _to_adams(
        self, operands: set[Operand], group_context: GrothGroupContext
    ) -> sp.Expr:
        """
        Converts this subtree into an equivalent adams polynomial.
        """
        raise NotImplementedError("This method must be implemented in all subclasses.")

    def _to_adams_lambda(
        self,
        operands: set[Operand],
        group_context: GrothGroupContext,
        adams_degree: int = 1,
    ) -> sp.Expr:
        """
        Converts this subtree into an equivalent adams polynomial.
        """
        return self._to_adams(operands, group_context)

    def _subs_adams(self, group_context: GrothGroupContext, ph: sp.Expr) -> sp.Expr:
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


class Variable(Operand):
    """
    An operand node in an expression tree that represents an abstract variable.

    Parameters:
    -----------
    value : sympy.Symbol
        The value of the variable, i.e. how it is represented.
    parent : Node
        The parent node of the variable. If the variable is the root
        of the tree, parent is None.
    id_ : hashable
        The id of the variable, used to identify it (if two operands
        have the same id, they are the same operand).

    Methods:
    --------
    sigma(degree: int) -> ET
        Applies the sigma operation to the current variable.
    lambda_(degree: int) -> ET
        Applies the lambda operation to the current variable.
    adams(degree: int) -> ET
        Applies the adams operation to the current variable.

    Properties:
    -----------
    sympy : sp.Expr
        The sympy representation of the node (and any children).
    """

    def __init__(
        self,
        value: sp.Symbol | str,
        parent: Optional[Node] = None,
        id_: Hashable = None,
    ):
        if isinstance(value, str):
            value = sp.Symbol(value)
        self.value: sp.Symbol = value

        super().__init__(parent, id_)
        self._adams_vars: list[sp.Symbol] = [1, self.value]
        self._lambda_vars: list[sp.Symbol] = [1, self.value]
        self._generate_adams_vars(1)

    def __repr__(self) -> str:
        return str(self.value)

    def _generate_adams_vars(self, n: int) -> None:
        """
        Generate the adams variables needed up to degree n.
        """
        self._adams_vars += [
            sp.Symbol(f"ψ_{i}({self.value})")
            for i in range(len(self._adams_vars), n + 1)
        ]

    def _generate_lambda_vars(self, n: int) -> None:
        """
        Generate the lambda variables needed up to degree n.
        """
        self._lambda_vars += [
            sp.Symbol(f"λ_{i}({self.value})")
            for i in range(len(self._lambda_vars), n + 1)
        ]

    def get_adams_var(self, i: int) -> sp.Symbol:
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
        self._generate_adams_vars(i)
        return self._adams_vars[i]

    def get_lambda_var(self, i: int) -> sp.Symbol:
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
        self._generate_lambda_vars(i)
        return self._lambda_vars[i]

    @dispatch(int, int)
    def _to_adams(self, degree: int, ph: int) -> int:
        """
        Catches the case where the polynomial is an integer.
        """
        return ph

    @dispatch(int, sp.Expr)
    def _to_adams(self, degree: int, ph: sp.Expr) -> sp.Expr:
        """
        Applies an adams operation of degree `degree` to any instances of this variable
        in the polynomial `ph`.
        """
        max_adams = -1
        operands = ph.free_symbols
        for i, adams in reversed(list(enumerate(self._adams_vars))):
            if adams in operands:
                max_adams = i
                break

        return ph.xreplace(
            {
                self.get_adams_var(i): self.get_adams_var(degree * i)
                for i in range(1, max_adams + 1)
            }
        )

    @dispatch(set, GrothGroupContext)
    def _to_adams(
        self, operands: set[Operand], group_context: GrothGroupContext
    ) -> sp.Expr:
        """
        Returns the adams of degree 1 of the variable.
        """
        return self.get_adams_var(1)

    def _subs_adams(self, group_context: GrothGroupContext, ph: sp.Expr) -> sp.Expr:
        """
        Substitutes any instances of an adams of this variable (ψ_d(variable)) into its
        equivalent polynomial of lambdas.
        """
        ph = ph.xreplace(
            {
                self.get_adams_var(i): group_context.get_lambda_2_adams_pol(i)
                for i in range(2, len(self._adams_vars))
            }
        )
        ph = ph.xreplace(
            {
                group_context.lambda_vars[i]: self.get_lambda_var(i)
                for i in range(1, len(group_context.lambda_vars))
            }
        )
        return ph

    @property
    def sympy(self) -> sp.Symbol:
        """
        The sympy representation of the variable.
        """
        return self.value


class Integer(Operand):
    """
    An operand in an expression tree that represents an integer.

    Parameters:
    -----------
    value : int
        The value of the integer.
    parent : Node
        The parent node of the integer.
    id_ : hashable
        The id of the integer, used to identify it (if two operands
        have the same id, they are the same operand).

    Methods:
    --------
    sigma(degree: int) -> ET
        Applies the sigma operation to the current integer.
    lambda_(degree: int) -> ET
        Applies the lambda operation to the current integer.
    adams(degree: int) -> ET
        Applies the adams operation to the current integer.

    Properties:
    -----------
    sympy : sp.Integer
        The sympy representation of the integer.
    """

    def __init__(
        self, value: int = 1, parent: Optional[Node] = None, id_: Hashable = None
    ):
        self.value: int = value
        super().__init__(parent, id_)

    def __repr__(self) -> str:
        return str(self.value)

    def get_adams_var(self, i: int) -> int:
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
        return self.value

    def get_lambda_var(self, i: int) -> int:
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
        return math.comb(i + self.value - 1, i)

    @dispatch(int, object)
    def _to_adams(self, degree: int, ph: sp.Expr) -> sp.Expr:
        """
        Applies an adams operation of degree `degree` to any instances of this integer
        in the polynomial `ph`.
        """
        return ph

    @dispatch(set, GrothGroupContext)
    def _to_adams(
        self, operands: set[Operand], group_context: GrothGroupContext
    ) -> sp.Expr:
        """
        Returns the integer.
        """
        return sp.Integer(self.value)

    def _subs_adams(self, group_context: GrothGroupContext, ph: sp.Expr) -> sp.Expr:
        """
        Substitutes any instances of an adams of this integer into its equivalent
        polynomial of lambdas.
        """
        return ph

    @property
    def sympy(self) -> sp.Integer:
        """
        The sympy representation of the integer.
        """
        return sp.Integer(self.value)


class Motive(Operand):
    """
    An abstract operand node in an expression tree that represents a motive.

    Parameters:
    -----------
    parent : Node
        The parent node of the motive.
    id_ : hashable
        The id of the motive, used to identify it (if two operands
        have the same id, they are the same operand).

    Methods:
    --------
    sigma(degree: int) -> ET
        Applies the sigma operation to the current motive.
    lambda_(degree: int) -> ET
        Applies the lambda operation to the current motive.
    adams(degree: int) -> ET
        Applies the adams operation to the current motive.

    Properties:
    -----------
    sympy : sp.Symbol
        The sympy representation of the motive.
    """

    def __init__(self, parent: Optional[Node] = None, id_: Hashable = None):
        super().__init__(parent, id_)


class Point(Motive):
    """
    An operand node in an expression tree that represents a point motive.

    Parameters:
    -----------
    parent : Node
        The parent node of the point motive.
    id_ : hashable
        Because the point motive is unique, the id is always "point". It should
        not be changed.

    Methods:
    --------
    sigma(degree: int) -> ET
        Applies the sigma operation to the current point motive.
    lambda_(degree: int) -> ET
        Applies the lambda operation to the current point motive.
    adams(degree: int) -> ET
        Applies the adams operation to the current point motive.

    Properties:
    sympy : sp.Symbol
        The sympy representation of the point.
    """

    def __init__(self, parent: Optional[Node] = None, id_: Hashable = "point"):
        id_ = "point"
        super().__init__(parent, id_)

    def __repr__(self) -> str:
        return "pt"

    def get_adams_var(self, i: int) -> int:
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
        return 1

    def get_lambda_var(self, i: int) -> int:
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
        return 1

    @dispatch(int, object)
    def _to_adams(self, degree: int, ph: sp.Expr) -> sp.Expr:
        """
        Applies an adams operation of degree `degree` to any instances of this point
        in the polynomial `ph`.
        """
        return ph

    @dispatch(set, GrothGroupContext)
    def _to_adams(
        self, operands: set[Operand], group_context: GrothGroupContext
    ) -> sp.Expr:
        """
        Returns the adams of degree 1 of the point.
        """
        return sp.Integer(1)

    def _subs_adams(self, group_context: GrothGroupContext, ph: sp.Expr) -> sp.Expr:
        """
        Substitutes any instances of an adams of this point into its equivalent
        polynomial of lambdas.
        """
        return ph

    @property
    def sympy(self) -> sp.Symbol:
        """
        The sympy representation of the point motive.
        """
        return sp.Symbol("pt")


class Symbol(Motive):
    """
    An operand in an expression tree that represents an abstract one
    dimensional motive, like the Lefschetz motive.

    Parameters:
    -----------
    value : sympy.Symbol
        The value of the symbolic integer, i.e. how it is represented.
    parent : Node
        The parent node of the symbolic integer.
    id_ : hashable
        The id of the symbolic integer, used to identify it (if two operands
        have the same id, they are the same operand).

    Methods:
    --------
    sigma(degree: int) -> ET
        Applies the sigma operation to the current symbolic integer.
    lambda_(degree: int) -> ET
        Applies the lambda operation to the current symbolic integer.
    adams(degree: int) -> ET
        Applies the adams operation to the current symbolic integer.

    Properties:
    -----------
    sympy : sp.Symbol
        The sympy representation of the symbol.
    """

    def __init__(
        self,
        value: sp.Symbol | str,
        parent: Optional[Node] = None,
        id_: Hashable = None,
    ):
        if isinstance(value, str):
            value = sp.Symbol(value)
        self.value: sp.Symbol = value
        super().__init__(parent, id_)

    def __repr__(self) -> str:
        return str(self.value)

    def get_adams_var(self, i: int) -> sp.Symbol:
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
        return self.value

    def get_lambda_var(self, i: int) -> sp.Expr:
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
        return self.value**i

    @dispatch(int, int)
    def _to_adams(self, degree: int, ph: int) -> int:
        """
        Catches the case where the polynomial is an integer.
        """
        return ph

    @dispatch(int, sp.Expr)
    def _to_adams(self, degree: int, ph: sp.Expr) -> sp.Expr:
        """
        Applies an adams operation of degree `degree` to any instances of this symbol
        in the polynomial `ph`.
        """
        return ph.xreplace({self.value: self.value**degree})

    @dispatch(set, GrothGroupContext)
    def _to_adams(
        self, operands: set[Operand], group_context: GrothGroupContext
    ) -> sp.Expr:
        """
        Returns the adams of degree 1 of the symbol.
        """
        return self.value

    def _subs_adams(self, group_context: GrothGroupContext, ph: sp.Expr) -> sp.Expr:
        """
        Substitutes any instances of an adams of this symbol into its equivalent
        polynomial of lambdas.
        """
        return ph

    @property
    def sympy(self) -> sp.Symbol:
        """
        The sympy representation of the symbol.
        """
        return sp.Symbol(f"s_{self.value}")


class Lefschetz(Symbol):
    """
    A symbol in an expression tree that represents a Lefschetz motive.

    Parameters:
    -----------
    parent : Node
        The parent node of the Lefschetz motive.
    id_ : hashable
        Because the Lefschetz motive is unique, the id is always "Lefschetz". It should
        not be changed.

    Methods:
    --------
    sigma(degree: int) -> ET
        Applies the sigma operation to the current Lefschetz motive.
    lambda_(degree: int) -> ET
        Applies the lambda operation to the current Lefschetz motive.
    adams(degree: int) -> ET
        Applies the adams operation to the current Lefschetz motive.

    Properties:
    -----------
    sympy : sp.Symbol
        The sympy representation of the Lefschetz motive.

    Attributes:
    -----------
    L_VAR : sympy.Symbol
        The value of the Lefschetz motive.
    """

    L_VAR: sp.Symbol = sp.Symbol("L")

    def __init__(self, parent=None, id_="Lefschetz"):
        id_ = "Lefschetz"
        super().__init__(self.L_VAR, parent, id_)


class CurveHodge(Motive):
    """
    An operand node in an expression tree that represents a Hodge motive.

    Parameters:
    -----------
    value : sympy.Symbol
        The value of the Hodge motive, i.e. how it is represented. It is
        the same as the value of its curve.
    g : int
        The genus of the Hodge motive. It is the same as the genus of its curve.
    parent : Node
        The parent node of the Hodge motive.
    id_ : hashable
        The id of the Hodge motive, used to identify it (if two operands
        have the same id, they are the same operand).

    Methods:
    --------
    sigma(degree: int) -> ET
        Applies the sigma operation to the current Hodge motive.
    lambda_(degree: int) -> ET
        Applies the lambda operation to the current Hodge motive.
    adams(degree: int) -> ET
        Applies the adams operation to the current Hodge motive.
    Z(t: Operand | int) -> ET
        Returns the generating function of the Hodge curve.

    Properties:
    -----------
    sympy : sp.Symbol
        The sympy representation of the Hodge motive.
    """

    def __init__(
        self, value: sp.Symbol | str, g: int = 1, parent=None, id_=None
    ) -> None:
        self.g: int = g
        if isinstance(value, str):
            value = sp.Symbol(value)
        self.value: sp.Symbol = value
        super().__init__(parent, id_)

        self.lambda_symbols: list[sp.Symbol] = [
            1,
            *[sp.Symbol(f"a{i}(h_{self.value})") for i in range(1, g + 1)],
        ]

        self._domain = sp.ZZ[[Lefschetz.L_VAR] + self.lambda_symbols[1:]]
        self._ring: PolyRing = self._domain.ring
        self._domain_symbols = [1] + [
            self._domain(var) for var in self.lambda_symbols[1:]
        ]
        self.lef_symbol = self._domain(Lefschetz.L_VAR)

        self._px_inv: list[sp.Expr] = [1]
        self._lambda_vars_pol: list[PolyElement] = [
            (
                self._domain_symbols[i]
                if i <= self.g
                else self._domain_symbols[2 * self.g - i]
                * self.lef_symbol ** (i - self.g)
            )
            for i in range(2 * self.g + 1)
        ]
        self._lambda_vars = [1] + [
            expr_from_pol(pol) for pol in self._lambda_vars_pol[1:]
        ]
        self._lambda_to_adams_pol: list[PolyElement] = [0]
        self.lambda_to_adams: list[sp.Expr] = [0]
        self._adams_vars: list[sp.Symbol] = [1, sp.Symbol(f"ψ_1(h_{self.value})")]

    def __repr__(self) -> str:
        return f"h_{self.value}"

    def _generate_adams_vars(self, n: int) -> None:
        """
        Generates the adams variables needed up to degree n.
        """
        self._adams_vars += [
            sp.Symbol(f"ψ_{i}(h_{self.value})")
            for i in range(len(self._adams_vars), n + 1)
        ]

    def _generate_lambda_vars(self, n: int) -> None:
        """
        Generates the lambda variables needed up to degree n.
        """
        self._lambda_vars_pol += [0] * (n - len(self._lambda_vars_pol) + 1)
        self._generate_inverse(n)

        for i in range(len(self._lambda_to_adams_pol), n + 1):
            self._lambda_to_adams_pol.append(
                self._ring.add(
                    *(
                        self._ring.mul(self._lambda_vars_pol[j], self._px_inv[i - j], j)
                        for j in range(1, i + 1)
                    )
                )
            )

    def get_adams_var(self, i: int) -> sp.Symbol:
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
        self._generate_adams_vars(i)
        return self._adams_vars[i]

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
        return 0 if i >= len(self._lambda_vars) else self._lambda_vars[i]

    @typechecked
    def Z(self, t: Operand | int | ET) -> ET:
        """
        Returns the generating function of the hodge curve.

        Parameters
        ----------
        t : Operand or int or ET
            The variable to use in the generating function.

        Returns
        -------
        The generating function of the hodge curve.
        """
        l = Lefschetz()

        et = ET(
            Add(
                [
                    (
                        self.lambda_(i) * t**i
                        if i <= self.g
                        else self.lambda_(2 * self.g - i) * l ** (i - self.g) * t**i
                    )
                    for i in range(2 * self.g + 1)
                ]
            )
        )

        return et

    @dispatch(int, int)
    def _to_adams(self, degree: int, ph: int) -> int:
        """
        Catches the case where the polynomial is an integer.
        """
        return ph

    @dispatch(int, sp.Expr)
    def _to_adams(self, degree: int, ph: sp.Expr) -> sp.Expr:
        """
        Applies an adams operation of degree `degree` to any instances of this hodge
        motive in the polynomial `ph`.
        """
        max_adams = -1
        operands = ph.free_symbols
        for i, adams in reversed(list(enumerate(self._adams_vars))):
            if adams in operands:
                max_adams = i
                break

        return ph.xreplace(
            {
                self.get_adams_var(i): self.get_adams_var(degree * i)
                for i in range(1, max_adams + 1)
            }
        )

    @dispatch(set, GrothGroupContext)
    def _to_adams(
        self, operands: set[Operand], group_context: GrothGroupContext
    ) -> sp.Expr:
        """
        Returns the adams of degree 1 of the hodge motive.
        """
        return self.get_adams_var(1)

    def _generate_inverse(self, n: int) -> None:
        """
        Fills the inverse list with the first n elements of the inverse Hodge curve.
        """
        for m in range(len(self._px_inv), n + 1):
            self._px_inv.append(
                -self._ring.add(
                    self._ring.mul(self._lambda_vars_pol[i], self._px_inv[m - i])
                    for i in range(1, m + 1)
                )
            )

    def _subs_adams(self, group_context: GrothGroupContext, ph: sp.Expr) -> sp.Expr:
        """
        Substitutes any instances of an adams of this hodge motive (ψ_d(hodge)) into its
        equivalent polynomial of lambdas.
        """
        # Find the maximum adams degree of this variable in the polynomial
        max_adams = -1
        operands = ph.free_symbols
        for i, adams in reversed(list(enumerate(self._adams_vars))):
            if adams in operands:
                max_adams = i
                break

        # Generate the lambda_to_adams polynomials up to the degree needed
        self._generate_lambda_vars(len(self._adams_vars) - 1)

        for i in range(len(self.lambda_to_adams), max_adams + 1):
            self.lambda_to_adams.append(expr_from_pol(self._lambda_to_adams_pol[i]))

        # Substitute the adams variables into the polynomial
        ph = ph.xreplace(
            {
                self.get_adams_var(i): self.lambda_to_adams[i]
                for i in range(1, max_adams + 1)
            }
        )

        return ph

    @property
    def sympy(self) -> sp.Symbol:
        """
        The sympy representation of the curve motive.
        """
        return sp.Symbol(f"h_{self.value}")


class Curve(Motive):
    """
    An operand in an expression tree that represents a curve.

    Parameters:
    -----------
    value : sympy.Symbol
        The value of the curve, i.e. how it is represented.
    g : int
        The genus of the curve.
    parent : Node
        The parent node of the curve.
    id_ : hashable
        The id of the curve, used to identify it (if two operands
        have the same id, they are the same operand).

    Methods:
    --------
    sigma(degree: int) -> ET
        Applies the sigma operation to the current curve.
    lambda_(degree: int) -> ET
        Applies the lambda operation to the current curve.
    adams(degree: int) -> ET
        Applies the adams operation to the current curve.
    P(t: Variable) -> ET
        Returns the generating function of the hodge curve.
    Z(t: Variable) -> ET
        Returns the generating function of the curve.

    Properties:
    -----------
    sympy : sp.Symbol
        The sympy representation of the curve.
    Jac : ET
        The Jacobian of the curve.
    """

    def __init__(
        self,
        value: sp.Symbol | str,
        g: int = 1,
        parent: Optional[Node] = None,
        id_: Hashable = None,
    ) -> None:
        self.g: int = g
        if isinstance(value, str):
            value = sp.Symbol(value)
        self.value: sp.Symbol = value
        super().__init__(parent, id_)

        self.point: Point = Point()
        self.curve_hodge: CurveHodge = CurveHodge(value, g, id_=self.id + "_hodge")
        self.lefschetz: Lefschetz = Lefschetz()

        self._ring: PolyRing = self.curve_hodge._domain.ring

        self.jac: sp.Expr | None = None
        self._lambda_vars: dict[int, sp.Expr] = {}
        self._lambda_vars_pol: list[PolyElement] = []
        self._adams_vars: list[sp.Symbol] = [1]

    def __repr__(self) -> str:
        return str(self.value)

    def _generate_adams_vars(self, n: int) -> None:
        """
        Generates the adams variables needed up to degree n.
        """
        self.curve_hodge._generate_adams_vars(n)
        self._adams_vars += [
            sp.Symbol(f"ψ_{i}(C_{self.value})")
            for i in range(len(self._adams_vars), n + 1)
        ]

    def _generate_lambda_vars(self, n: int) -> None:
        """
        Generates the lambda variables needed up to degree n.
        """
        self.curve_hodge._generate_lambda_vars(n)

        if len(self._lambda_vars_pol) == 0:
            self._lambda_vars_pol = [1]

        for i in range(len(self._lambda_vars_pol), n + 1):
            self._lambda_vars_pol.append(
                self._lambda_vars_pol[i - 1]
                + self._ring.add(
                    *(
                        self._ring.mul(
                            self.curve_hodge._lambda_vars_pol[i - j],
                            self.curve_hodge.lef_symbol**j,
                        )
                        for j in range(i + 1)
                    )
                )
            )

    def get_adams_var(self, i: int) -> sp.Symbol:
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
        self._generate_adams_vars(i)
        return self._adams_vars[i]

    def get_lambda_var(self, i: int) -> sp.Expr:
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
        self._generate_lambda_vars(i)

        if i not in self._lambda_vars:
            self._lambda_vars[i] = expr_from_pol(self._lambda_vars_pol[i])

        return self._lambda_vars[i]

    @typechecked
    def P(self, t: Operand | int | ET) -> ET:
        """
        Returns the generating function of the hodge curve.

        Parameters
        ----------
        t : Operand or int or ET
            The variable to use in the generating function.

        Returns
        -------
        The generating function of the hodge curve.
        """
        return self.curve_hodge.Z(t)

    @property
    def Jac(self) -> ET:
        """
        The Jacobian of the curve.
        """
        if self.jac is None:
            self.jac = self.curve_hodge.Z(1)
        return self.jac

    @typechecked
    def Z(self, t: Operand | int | ET) -> ET:
        """
        Returns the generating function of the curve.

        Parameters
        ----------
        t : Operand or int or ET
            The variable to use in the generating function.

        Returns
        -------
        The generating function of the curve.
        """
        return self.P(t) / ((1 - t) * (1 - self.lefschetz * t))

    @dispatch(int, int)
    def _to_adams(self, degree: int, ph: int) -> int:
        """
        Catches the case where the polynomial is an integer.
        """
        return ph

    @dispatch(int, sp.Expr)
    def _to_adams(self, degree: int, ph: sp.Expr) -> sp.Expr:
        """
        Applies an adams operation of degree `degree` to any instances of this curve
        in the polynomial `ph`.
        """
        ph = self.curve_hodge._to_adams(degree, ph)
        return ph

    @dispatch(set, GrothGroupContext)
    def _to_adams(
        self, operands: set[Operand], group_context: GrothGroupContext
    ) -> sp.Expr:
        """
        Returns the adams of degree 1 of this curve.
        """
        return (
            self.curve_hodge._to_adams(operands, group_context)
            + self.lefschetz._to_adams(operands, group_context)
            + self.point._to_adams(operands, group_context)
        )

    def _subs_adams(self, group_context: GrothGroupContext, ph: sp.Expr) -> sp.Expr:
        """
        Substitutes any instances of an adams of this curve (ψ_d(curve)) into its
        equivalent polynomial of lambdas.
        """
        ph = self.curve_hodge._subs_adams(group_context, ph)
        return ph

    @property
    def sympy(self) -> sp.Symbol:
        """
        The sympy representation of the curve.
        """
        return sp.Symbol(f"C_{self.value}_{self.g}")


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
        self, operands: set[Operand], group_context: GrothGroupContext
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
        group_context: GrothGroupContext,
        adams_degree: int = 1,
    ) -> sp.Expr:
        """
        Converts this subtree into an equivalent adams polynomial, when applying
        to_lambda. Keeps some lambda variables as they are.

        Parameters
        ----------
        operands : set[Operand]
            The operands in the tree.
        group_context : GrothGroupContext
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
        self, operands: set[Operand], group_context: GrothGroupContext
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
        group_context: GrothGroupContext,
        adams_degree: int = 1,
    ) -> sp.Expr:
        """
        Converts this subtree into an equivalent adams polynomial, when applying
        to_lambda. Keeps some lambda variables as they are.

        Parameters
        ----------
        operands : set[Operand]
            The operands in the tree.
        group_context : GrothGroupContext
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
        self, operands: set[Operand], group_context: GrothGroupContext
    ) -> sp.Expr:
        """
        Converts this subtree into an equivalent adams polynomial. It calls _to_adams
        on the child and returns the negative of the result.
        """
        return -self.child._to_adams(operands, group_context)

    def _to_adams_lambda(
        self,
        operands: set[Operand],
        group_context: GrothGroupContext,
        adams_degree: int = 1,
    ) -> sp.Expr:
        """
        Converts this subtree into an equivalent adams polynomial, when applying
        to_lambda. Keeps some lambda variables as they are.

        Parameters
        ----------
        operands : set[Operand]
            The operands in the tree.
        group_context : GrothGroupContext
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
        self, operands: set[Operand], group_context: GrothGroupContext
    ) -> sp.Expr:
        """
        Converts this subtree into an equivalent adams polynomial. It calls _to_adams
        on the child and returns the inverse of the result.
        """
        return 1 / self.child._to_adams(operands, group_context)

    def _to_adams_lambda(
        self,
        operands: set[Operand],
        group_context: GrothGroupContext,
        adams_degree: int = 1,
    ) -> sp.Expr:
        """
        Converts this subtree into an equivalent adams polynomial, when applying
        to_lambda. Keeps some lambda variables as they are.

        Parameters
        ----------
        operands : set[Operand]
            The operands in the tree.
        group_context : GrothGroupContext
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
        self, operands: set[Operand], group_context: GrothGroupContext
    ) -> sp.Expr:
        """
        Converts this subtree into an equivalent adams polynomial. It calls _to_adams
        on the child and returns the result raised to the power of the degree.
        """
        return self.child._to_adams(operands, group_context) ** self.degree

    def _to_adams_lambda(
        self,
        operands: set[Operand],
        group_context: GrothGroupContext,
        adams_degree: int = 1,
    ) -> sp.Expr:
        """
        Converts this subtree into an equivalent adams polynomial, when applying
        to_lambda. Keeps some lambda variables as they are.

        Parameters
        ----------
        operands : set[Operand]
            The operands in the tree.
        group_context : GrothGroupContext
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
        self, operands: set[Operand], group_context: GrothGroupContext
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
        group_context: GrothGroupContext,
        adams_degree: int = 1,
    ) -> sp.Expr:
        """
        Converts this subtree into an equivalent adams polynomial, when applying
        to_lambda. Keeps some lambda variables as they are.

        Parameters
        ----------
        operands : set[Operand]
            The operands in the tree.
        group_context : GrothGroupContext
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
        self, operands: set[Operand], group_context: GrothGroupContext
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
        group_context: GrothGroupContext,
        adams_degree: int = 1,
    ) -> sp.Expr:
        """
        Converts this subtree into an equivalent adams polynomial, when applying
        to_lambda. Keeps some lambda variables as they are.

        Parameters
        ----------
        operands : set[Operand]
            The operands in the tree.
        group_context : GrothGroupContext
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
        self, operands: set[Operand], group_context: GrothGroupContext
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
        group_context: GrothGroupContext,
        adams_degree: int = 1,
    ) -> sp.Expr:
        """
        Converts this subtree into an equivalent adams polynomial, when applying
        to_lambda. Keeps some lambda variables as they are.

        Parameters
        ----------
        operands : set[Operand]
            The operands in the tree.
        group_context : GrothGroupContext
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

    def to_adams(self, group_context: GrothGroupContext = None) -> sp.Expr:
        """
        Turns the current tree into a polynomial of adams operators.

        Parameters
        ----------
        group_context : GrothGroupContext
            The group context of the tree. If None, a new one is created.

        Returns
        -------
        The polynomial of adams operators equivalent to the current tree.
        """
        # Initialize a GrothGroupContext
        if group_context is None:
            group_context = GrothGroupContext()

        return self.root._to_adams(self.operands, group_context)

    def to_lambda(
        self, group_context: GrothGroupContext = None, *, optimize=True
    ) -> sp.Expr:
        """
        Turns the current tree into a polynomial of lambda operators.

        Parameters
        ----------
        group_context : GrothGroupContext
            The group context of the tree. If None, a new one is created.
        optimize : bool
            If True, optimizations are implemented when converting to lambda.

        Returns
        -------
        The polynomial of lambda operators equivalent to the current tree.
        """
        # Initialize a GrothGroupContext
        if group_context is None:
            group_context = GrothGroupContext()

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
                return ET(CurveHodge(sp.Symbol(expr.name[2:])))
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
