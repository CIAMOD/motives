import sympy as sp

from ..core import LambdaRingContext
from .motive import Motive
from .lefschetz import Lefschetz


class Proj(Motive, sp.AtomicExpr):
    """
    Represents the motivic class of a complex projective space P^n in the Grothendieck lambda-ring of
    varieties, the Grothendieck ring of Chow motives, or any extension or completion of such rings.

    It is equal to the sum
        1 + L + ... + L^n,
    where L is the Lefschetz motive. This class supports Adams and lambda operations.

    Attributes:
    -----------
    n : int
        The dimension of the projective space.
    lef : Lefschetz
        The Lefschetz motive used in the motive of the projective space.
    _et_repr : sp.Expr
        The projective space motive as a SymPy expression.
    _lambda_vars : dict[int, sp.Expr]
        A dictionary of lambda variables generated for this projective space.
    """

    def __new__(cls, n: int, *args, **kwargs):
        """
        Creates a new instance of `Proj`.

        Args:
        -----
        n : int
            The dimension of the projective space.

        Returns:
        --------
        Proj
            A new instance of the `Proj` class.
        """
        new_proj = sp.AtomicExpr.__new__(cls)
        new_proj._assumptions["commutative"] = True
        return new_proj

    def __init__(self, n: int, *args, **kwargs) -> None:
        """
        Initializes a `Proj` instance.

        Args:
        -----
        n : int
            The dimension of the projective space.
        """
        self.lef: Lefschetz = Lefschetz()
        self.n: int = n
        self._et_repr: sp.Expr = sp.Add(*[self.lef**j for j in range(self.n + 1)])
        self._lambda_vars: dict[int, sp.Expr] = {}

    def __repr__(self) -> str:
        """
        Returns the string representation of the projective space.

        Returns:
        --------
        str
            The string representation in the form "P^n".
        """
        return f"P^{self.n}"

    def _hashable_content(self) -> tuple:
        """
        Returns the hashable content of the projective space.

        Returns:
        --------
        tuple
            A tuple containing the dimension of the projective space.
        """
        return (self.n,)

    def get_adams_var(self, i: int, as_symbol: bool = False) -> sp.Expr:
        """
        Returns the projective space with an Adams operation applied.

        The Adams operation on a projective space is equivalent to summing
        the powers of the Lefschetz motive raised to the i-th power.

        Args:
        -----
        i : int
            The degree of the Adams operator.
        as_symbol : bool, optional
            If True, returns the Adams variable as a SymPy Symbol.

        Returns:
        --------
        sp.Expr
            The projective space with the Adams operator applied.
        """
        return sp.Add(*[self.lef ** (j * i) for j in range(self.n + 1)])

    def get_lambda_var(self, i: int, as_symbol: bool = False) -> sp.Expr:
        """
        Returns the projective space with a Lambda operation applied to it.

        The Lambda operation is applied by converting Adams variables into
        Lambda variables using the context's transformation rules.

        Args:
        -----
        i : int
            The degree of the Lambda operator.
        as_symbol : bool, optional
            If True, returns the Lambda variable as a SymPy Symbol. Otherwise, returns it as a
            Lambda object.

        Returns:
        --------
        sp.Expr
            The projective space with the Lambda operator applied.
        """
        if i not in self._lambda_vars:
            lrc = LambdaRingContext()

            ph_list = [
                self.lef._apply_adams(
                    j,
                    self._et_repr,
                    0,  # The maximum Adams degree is not needed for the Lefschetz motive.
                    as_symbol=True,
                )
                for j in range(i + 1)
            ]

            self._lambda_vars[i] = lrc.get_adams_2_lambda_pol(i).xreplace(
                {lrc.adams_vars[i]: ph_list[i] for i in range(i + 1)}
            )

        return self._lambda_vars[i]

    def _apply_adams(
        self, degree: int, ph: sp.Expr, max_adams_degree: int, as_symbol: bool = False
    ) -> sp.Expr:
        """
        Applies the Adams operator to any instances of this projective space in the expression.

        This method raises an exception because projective spaces should not appear directly
        as leaves of the expression. Instead, they should be decomposed into their components.

        Args:
        -----
        degree : int
            The degree of the Adams operator to apply.
        ph : sp.Expr
            The polynomial in which to apply the Adams operator.

        Raises:
        -------
        Exception
            Raised because projective spaces should be decomposed into their components.
        """
        raise Exception(
            f"There is a projective space in the expression {ph}. "
            "It should have been converted to its components."
        )

    def _subs_adams(
        self, ph: sp.Expr, max_adams_degree: int, as_symbol: bool = False
    ) -> sp.Expr:
        """
        Substitutes Adams variables of this projective space in the expression with Lambda polynomials.

        Args:
        -----
        ph : sp.Expr
            The polynomial in which to substitute the Adams variables.

        Raises:
        -------
        Exception
            Raised because projective spaces should be decomposed into their components.
        """
        raise Exception(
            f"There is a projective space in the expression {ph}. "
            "It should have been converted to its components."
        )

    @property
    def free_symbols(self) -> set[sp.Symbol]:
        """
        Returns the set of free symbols in the projective space.

        Returns:
        --------
        set[sp.Symbol]
            The set of free symbols in the projective space.
        """
        return {self.lef}
