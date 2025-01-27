import sympy as sp
from ..motive import Motive
from typing import Iterable
from ..lefschetz import Lefschetz
from ...core import LambdaRingContext
from ...core.operand.operand import Operand


class SemisimpleG(Motive, sp.AtomicExpr):
    """
    Represents the Grothendieck motive of a connected semisimple simply connected algebraic
    group as a SymPy atomic expression.

    This class inherits from both Motive and sympy.AtomicExpr, and represents
    a Grothendieck motive with a given set of integers (ds) and a dimension.

    Attributes
    ----------
    ds : list[int]
        The set of integers used to construct the motive.
    dim : int
        The dimension of the motive.
    lef : Lefschetz
        The Lefschetz motive associated with this motive.
    _et_repr : sp.Expr
        The étale representation of the motive.
    _lambda_vars : dict[int, sp.Expr]
        A dictionary to store lambda variables.

    Methods
    -------
    __new__(cls, ds: list[int], dim: int, *args, **kwargs) -> 'SemisimpleG'
        Creates a new instance of the SemisimpleG class with the specified ds and dimension.
    __init__(self, ds: list[int], dim: int, *args, **kwargs) -> None
        Initializes the SemisimpleG class with the specified ds and dimension.
    BG(self) -> Motive
        Computes the motive of the classifying stack of G.
    get_adams_var(self, i: int, as_symbol: bool = False) -> sp.Expr
        Returns the Adams variable of the motive for a given index.
    get_lambda_var(self, i: int, as_symbol: bool = False) -> sp.Expr
        Returns the lambda variable of the motive for a given index.
    _apply_adams(self, degree: int, ph: sp.Expr, max_adams_degree: int, as_symbol: bool = False) -> sp.Expr
        Applies the Adams operator to the expression.
    _subs_adams(self, ph: sp.Expr, max_adams_degree: int, as_symbol: bool = False) -> sp.Expr
        Substitutes Adams variables in the expression with Lambda polynomials.
    free_symbols(self) -> set[sp.Symbol]
        Returns the set of free symbols in the group, including the Lefschetz motive.
    """

    def __new__(cls, ds: list[int], dim: int, *args, **kwargs) -> "SemisimpleG":
        """
        Creates a new instance of the SemisimpleG class with the specified ds and dimension.

        Parameters
        ----------
        ds : list[int]
            The set of integers used to construct the motive.
        dim : int
            The dimension of the motive.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        SemisimpleG
            A new instance of the SemisimpleG class.

        Raises
        ------
        ValueError
            If ds is not a list.
        """
        if not isinstance(ds, list):
            raise ValueError("The set of integers ds must be a list.")
        new_g = sp.AtomicExpr.__new__(cls)
        new_g._assumptions["commutative"] = True
        return new_g

    def __init__(self, ds: list[int], dim: int, *args, **kwargs) -> None:
        """
        Initializes the SemisimpleG class with the specified ds and dimension.

        Parameters
        ----------
        ds : list[int]
            The set of integers used to construct the motive.
        dim : int
            The dimension of the motive.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        self.ds: list[int] = ds
        self.rk: int = len(ds)
        self.dim: int = 2 * sum(ds) - self.rk

        self.lef: Lefschetz = Lefschetz()
        self._et_repr: sp.Expr = self.lef**self.dim * sp.Mul(
            *(1 - self.lef ** (-d) for d in ds)
        )
        self._lambda_vars: dict[int, sp.Expr] = {}

    def BG(self) -> Motive:
        """
        Computes the motive of the classifying stack of G, BG = [pt/G], taking
        [BG] = 1/[G].

        Returns
        -------
        Motive
            The motivic class of the classifying stack BG of G.
        """
        return 1 / self

    def get_adams_var(self, i: int, as_symbol: bool = False) -> sp.Expr:
        """
        Returns the Adams variable of the motive for a given index.

        Parameters
        ----------
        i : int
            The index of the Adams variable.
        as_symbol : bool, optional
            If True, returns the Adams variable as a SymPy Symbol. Otherwise, returns it as a
            SymPy expression.

        Returns
        -------
        sp.Expr
            The Adams variable of the motive.
        """
        return self.lef._apply_adams(i, self._et_repr, 0, as_symbol)

    def get_lambda_var(self, i: int, as_symbol: bool = False) -> sp.Expr:
        """
        Returns the lambda variable of the motive for a given index.

        Parameters
        ----------
        i : int
            The index of the lambda variable.
        as_symbol : bool, optional
            If True, returns the Lambda variable as a SymPy Symbol. Otherwise, returns it as a
            Lambda polynomial.

        Returns
        -------
        sp.Expr
            The lambda variable of the motive.
        """
        if i not in self._lambda_vars:
            lrc: LambdaRingContext = LambdaRingContext()

            ph_list: list[sp.Expr] = [
                self.get_adams_var(j, as_symbol) for j in range(i + 1)
            ]

            replace_dict: dict[sp.Expr, sp.Expr] = {
                lrc.adams_vars[j]: ph_list[j] for j in range(i + 1)
            }
            self._lambda_vars[i] = lrc.get_adams_2_lambda_pol(i).xreplace(replace_dict)

        return self._lambda_vars[i]

    def _apply_adams(
        self, degree: int, ph: sp.Expr, max_adams_degree: int, as_symbol: bool = False
    ) -> sp.Expr:
        """
        Applies the Adams operator to the expression.

        This method raises an exception because groups should not appear directly
        in the expression. Instead, they should be decomposed into their components.

        Parameters
        ----------
        degree : int
            The degree of the Adams operator to apply.
        ph : sp.Expr
            The polynomial in which to apply the Adams operator.
        max_adams_degree : int
            The maximum degree of Adams operations.
        as_symbol : bool, optional
            If True, returns symbols instead of expressions.

        Raises
        ------
        Exception
            Always raised as groups should be decomposed into their components.
        """
        raise Exception(
            f"There is a group in the expression {ph}. "
            "It should have been converted to its components."
        )

    def _subs_adams(
        self, ph: sp.Expr, max_adams_degree: int, as_symbol: bool = False
    ) -> sp.Expr:
        """
        Substitutes Adams variables in the expression with their equivalent Lambda polynomials.

        This method raises an exception because groups should not appear
        directly in the expression. They should be decomposed into their components.

        Parameters
        ----------
        ph : sp.Expr
            The polynomial in which to substitute the Adams variables.
        max_adams_degree : int
            The maximum degree of Adams operators in the expression.
        as_symbol : bool, optional
            If True, represents adams operators as symbols.

        Raises
        ------
        Exception
            Always raised as groups should be decomposed into their components.
        """
        raise Exception(
            f"There is a group in the expression {ph}. "
            "It should have been converted to its components."
        )

    @property
    def free_symbols(self) -> set[sp.Symbol]:
        """
        Returns the set of free symbols in the group, including the Lefschetz motive.

        Returns
        -------
        set[sp.Symbol]
            The set of free symbols in the group.
        """
        return {self.lef}
