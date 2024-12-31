import sympy as sp
from ..motive import Motive
from typing import Iterable
from ..lefschetz import Lefschetz
from ...core import LambdaRingContext
from ...core.operand.operand import Operand


# TODO revisar docs


class G(Motive, sp.AtomicExpr):
    """
    Represents a Grothendieck motive as a sympy atomic expression.

    This class inherits from both Motive and sympy.AtomicExpr, and represents
    a Grothendieck motive with a given set of integers (ds) and a dimension.

    Attributes:
    -----------
    ds : Iterable[int]
        The set of integers used to construct the motive.
    dim : int
        The dimension of the motive.
    lef : Lefschetz
        The Lefschetz motive associated with this motive.
    _et_repr : sp.Expr
        The etale representation of the motive.
    _lambda_vars : dict
        A dictionary to store lambda variables.

    Methods:
    --------
    __new__(cls, ds: Iterable[int], dim: int, *args, **kwargs)
        Creates a new instance of the G class with the specified ds and dimension.
    __init__(self, ds: Iterable[int], dim: int, *args, **kwargs)
        Initializes the G class with the specified ds and dimension.
    get_adams_var(self, i: int) -> sp.Expr
        Returns the Adams variable of the motive for a given index.
    get_lambda_var(self, i: int) -> sp.Expr
        Returns the lambda variable of the motive for a given index.
    _to_adams(self, degree: int, ph: sp.Expr) -> sp.Expr
        Applies the Adams operator to any instances of this group in the expression.
    _to_adams(self, operands: set[Operand]) -> sp.Expr
        Converts this group into an equivalent Adams polynomial.
    _subs_adams(self, ph: sp.Expr) -> sp.Expr
        Substitutes Adams variables of this group in the expression with their equivalent Lambda polynomials.
    free_symbols(self) -> set[sp.Symbol]
        Returns the set of free symbols in the group, which includes the Lefschetz motive.
    """

    def __new__(cls, ds: Iterable[int], dim: int, *args, **kwargs):
        """
        Creates a new instance of the G class with the specified ds and dimension.

        Parameters:
        -----------
        ds : Iterable[int]
            The set of integers used to construct the motive.
        dim : int
            The dimension of the motive.
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns:
        --------
        G
            A new instance of the G class.
        """
        new_g = sp.AtomicExpr.__new__(cls)
        new_g._assumptions["commutative"] = True
        return new_g

    def __init__(self, ds: Iterable[int], dim: int, *args, **kwargs):
        """
        Initializes the G class with the specified ds and dimension.

        Parameters:
        -----------
        ds : Iterable[int]
            The set of integers used to construct the motive.
        dim : int
            The dimension of the motive.
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        self.ds = ds
        self.dim = dim

        self.lef = Lefschetz()
        self._et_repr = self.lef**dim * sp.Mul(
            *[1 - self.lef ** (-d) for d in ds],
        )
        self._lambda_vars = {}

    def get_adams_var(self, i: int) -> sp.Expr:
        """
        Returns the Adams variable of the motive for a given index.

        Parameters:
        -----------
        i : int
            The index of the Adams variable.

        Returns:
        --------
        sp.Expr
            The Adams variable of the motive.
        """
        return self.lef._apply_adams(i, self._et_repr)

    def get_lambda_var(self, i: int) -> sp.Expr:
        """
        Returns the lambda variable of the motive for a given index.

        Parameters:
        -----------
        i : int
            The index of the lambda variable.

        Returns:
        --------
        sp.Expr
            The lambda variable of the motive.
        """
        if i not in self._lambda_vars:
            lrc = LambdaRingContext()

            ph_list = [self.get_adams_var(j) for j in range(i + 1)]

            self._lambda_vars[i] = lrc.get_adams_2_lambda_pol(i).xreplace(
                {lrc.adams_vars[i]: ph_list[i] for i in range(i + 1)}
            )

        return self._lambda_vars[i]

    def _apply_adams(self, degree: int, ph: sp.Expr) -> sp.Expr:
        """
        Applies the Adams operator to any instances of this group in the expression.

        This method raises an exception because groups should not appear directly
        in the expression. Instead, they should be decomposed into their components.

        Parameters:
        -----------
        degree : int
            The degree of the Adams operator to apply.
        ph : sp.Expr
            The polynomial in which to apply the Adams operator.

        Raises:
        -------
        Exception
            Always raised as groups should be decomposed into their components.
        """
        raise Exception(
            f"There is a group in the expression {ph}. "
            "It should have been converted to its components."
        )

    def _to_adams(self, operands: set[Operand]) -> sp.Expr:
        """
        Converts this group into an equivalent Adams polynomial.

        Parameters:
        -----------
        operands : set[Operand]
            The set of all operands in the expression tree.

        Returns:
        --------
        sp.Expr
            The Adams polynomial equivalent to this group.
        """
        return self._et_repr

    def _subs_adams(self, ph: sp.Expr) -> sp.Expr:
        """
        Substitutes Adams variables of this group in the expression
        with their equivalent Lambda polynomials.

        This method raises an exception because groups should not appear
        directly in the expression. They should be decomposed into their components.

        Parameters:
        -----------
        ph : sp.Expr
            The polynomial in which to substitute the Adams variables.

        Raises:
        -------
        Exception
            Always raised as groups should be decomposed into their components.
        """
        raise Exception(
            f"There is a group in the expression {ph}. "
            "It should have been converted to its components."
        )

    @property
    def free_symbols(self):
        """
        Returns the set of free symbols in the group, which includes
        the Lefschetz motive.

        Returns:
        --------
        set[sp.Symbol]
            The set of free symbols in the group.
        """
        return {self.lef}
