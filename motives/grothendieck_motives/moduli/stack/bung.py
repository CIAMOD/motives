from ...motive import Motive
from ...groups.semisimple_g import SemisimpleG
from ...curves.curve import Curve
from ...lefschetz import Lefschetz
import sympy as sp


class BunG(Motive, sp.AtomicExpr):
    """
    Represents the Grothendieck motivic class of the moduli stack of principal G-bundles on
    a smooth complex projective curve.

    The `BunG` is a motive associated with a `Curve` object and a semisimple group `G`.
    It supports Adams and lambda operations, generating functions, and interacts with other motives.

    Attributes:
    -----------
    curve : Curve
        The curve for which this BunG is defined.
    group : SemisimpleG
        The semisimple group associated with this BunG.
    lef : Lefschetz
        The Lefschetz motive.
    """

    def __new__(cls, curve: Curve, group: SemisimpleG, *args, **kwargs) -> "BunG":
        """
        Creates a new instance of the `BunG` class.

        Parameters:
        -----------
        curve : Curve
            The curve for which to create the BunG.
        group : SemisimpleG
            The semisimple group.

        Returns:
        --------
        BunG
            A new instance of the `BunG` class.
        """
        new_bun = sp.AtomicExpr.__new__(cls)
        new_bun._assumptions["commutative"] = True
        return new_bun

    def __init__(self, curve: Curve, group: SemisimpleG, *args, **kwargs) -> None:
        """
        Initializes a `BunG` instance.

        Parameters:
        -----------
        curve : Curve
            The curve for which this BunG is defined.
        group : SemisimpleG
            The semisimple group.
        """
        self.curve: Curve = curve
        self.group: SemisimpleG = group
        self.lef: Lefschetz = Lefschetz()

        self._et_repr: sp.Expr = self.lef ** (
            (self.curve.g - 1) * self.group.dim
        ) * sp.Mul(*[self.curve.Z(self.lef ** (-d)) for d in self.group.ds])

        self._lambda_vars: dict[int, sp.Expr] = {}
        self._adams_vars: dict[int, sp.Expr] = {}

    def __repr__(self) -> str:
        """
        Returns the string representation of the BunG.

        Returns:
        --------
        str
            A string representation in the form of "BunG_{curve}_{group}".
        """
        return f"BunG_{self.curve}_{self.group}"

    def _hashable_content(self) -> tuple:
        """
        Returns the hashable content of the BunG.

        Returns:
        --------
        tuple
            A tuple containing the curve and the semisimple group.
        """
        return (self.curve, self.group)

    @property
    def free_symbols(self) -> set:
        """
        Returns the set of free symbols in the BunG.

        Returns:
        --------
        set
            The set of free symbols in the BunG.
        """
        return self._et_repr.free_symbols

    def get_max_adams_degree(self) -> int:
        """
        Returns the maximum degree of the Adams operator for this BunG.

        Returns:
        --------
        int
            The maximum Adams degree.
        """
        return self._et_repr.get_max_adams_degree()

    def get_adams_var(self, i: int, as_symbol: bool = False) -> sp.Expr:
        """
        Returns the BunG with an Adams operation applied to it.

        Parameters:
        -----------
        i : int
            The degree of the Adams operator.
        as_symbol : bool, optional
            If True, returns the Adams variable as a SymPy Symbol. Otherwise, returns it as an
            Adams_ object.

        Returns:
        --------
        sp.Expr
            The BunG with the Adams operator applied.
        """
        if i not in self._adams_vars:
            self._adams_vars[i] = self._et_repr.adams(i).to_adams(as_symbol=True)

        if as_symbol is False:
            return self._adams_vars[i].xreplace(
                {
                    self.curve.curve_chow.get_adams_var(
                        j, as_symbol=True
                    ): self.curve.curve_chow.get_adams_var(j, as_symbol=False)
                    for j in range(2, i * self.curve.g + 1)
                }
            )

        return self._adams_vars[i]

    def get_lambda_var(self, i: int, as_symbol: bool = False) -> sp.Expr:
        """
        Returns the BunG with a Lambda operation applied to it.

        Args:
        -----
        i : int
            The degree of the Lambda operator.
        as_symbol : bool, optional
            If True, returns the Lambda variable as a SymPy Symbol. Otherwise, returns it as a
            Lambda_ object.

        Returns:
        --------
        sp.Expr
            The BunG with the Lambda operator applied.
        """
        if i not in self._lambda_vars:
            self._lambda_vars[i] = self._et_repr.lambda_(i).to_lambda(as_symbol=True)

        if not as_symbol:
            return self._lambda_vars[i].xreplace(
                {
                    symbol: self.curve.curve_chow.get_lambda_var(j, as_symbol=False)
                    for j, symbol in enumerate(
                        self.curve.curve_chow.lambda_symbols[2:], start=2
                    )
                }
            )

        return self._lambda_vars[i]

    def _apply_adams(
        self, degree: int, ph: sp.Expr, max_adams_degree: int, as_symbol: bool = False
    ) -> sp.Expr:
        """
        Applies the Adams operator to any instances of this BunG in the expression.

        Parameters:
        -----------
        degree : int
            The degree of the Adams operator to apply.
        ph : sp.Expr
            The polynomial in which to apply the Adams operator.
        max_adams_degree : int
            The maximum degree for Adams operations.
        as_symbol : bool, optional
            Whether to return the result as a symbol.

        Returns:
        --------
        sp.Expr
            The polynomial with the Adams operator applied.

        Raises:
        -------
        Exception
            Raised as BunG instances are found in the expression.
        """
        raise Exception(
            f"There is a BunG in the expression {ph}. "
            "It should have been converted to its components."
        )

    def _subs_adams(
        self, ph: sp.Expr, max_adams_degree: int, as_symbol: bool = False
    ) -> sp.Expr:
        """
        Substitutes Adams variables in the polynomial with equivalent Lambda polynomials.

        Parameters:
        -----------
        ph : sp.Expr
            The polynomial in which to substitute the Adams variables.
        max_adams_degree : int
            The maximum degree for Adams operations.
        as_symbol : bool, optional
            Whether to return the result as symbols.

        Returns:
        --------
        sp.Expr
            The polynomial with Adams variables substituted by Lambda polynomials.

        Raises:
        -------
        Exception
            Raised as BunG instances are found in the expression.
        """
        raise Exception(
            f"There is a BunG in the expression {ph}. "
            "It should have been converted to its components."
        )
