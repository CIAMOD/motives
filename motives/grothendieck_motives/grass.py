import sympy as sp
import numpy as np

from ..core import LambdaRingContext
from .motive import Motive
from .lefschetz import Lefschetz


class Grass(Motive, sp.AtomicExpr):
    """
    Represents the motivic class of a Grassmanian in the Grothendieck lambda-ring of
    varieties, the Grothendieck ring of Chow motives, or any extension or completion of such rings.

    Attributes:
    -----------
    k: int
        The dimension of the subpaces parametricized by the Grassmanian.
    n : int
        The dimension of vector space containing all subspaces.
    lef : Lefschetz
        The Lefschetz motive used in the motive of the projective space.
    exponents : List[int]
        Powers of Lefschetz to be summed to obtain the expression of the Grassmanian
    _et_repr : sp.Expr
        The projective space motive as a SymPy expression.
    _lambda_vars : dict[int, sp.Expr]
        A dictionary of lambda variables generated for this projective space.
    """

    def __new__(cls,k: int, n: int, *args, **kwargs):
        """
        Creates a new instance of `Grass`.

        Args:
        -----
        k: int
            The dimension of the subpaces parametricized by the Grassmanian.
        n : int
            The dimension of vector space containing all subspaces.

        Returns:
        --------
        Grass
            A new instance of the `Grass` class.
        """
        new_proj = sp.AtomicExpr.__new__(cls)
        new_proj._assumptions["commutative"] = True
        return new_proj

    def __init__(self, k: int, n: int) -> None:
        """
        Initializes a `Grass` instance.

        Args:
        -----
        k: int
            The dimension of the subpaces parametricized by the Grassmanian.
        n : int
            The dimension of vector space containing all subspaces.
        """
        self.lef: Lefschetz = Lefschetz()
        self.k: int = k
        self.n: int = n
        self.exponents = np.array(Grass._get_partitions(k, n-k)).sum(axis=1)
        self._et_repr: sp.Expr = sp.Add(*[self.lef**exp for exp in self.exponents])
        self._lambda_vars: dict[int, sp.Expr] = {}

    def __repr__(self) -> str:
        """
        Returns the string representation of the Grasmanian.

        Returns:
        --------
        str
            The string representation in the form "Grass(k, n)".
        """
        return f"Grass({self.k}, {self.n})"
    
    def _hashable_content(self) -> tuple:
        """
        Returns the hashable content of the Grassmanian.

        Returns:
        --------
        tuple
            A tuple containing the parameters of the Grassmanian.
        """
        return (self.k, self.n)
    
    def get_adams_var(self, i, as_symbol = False):
        """
        Returns the Grassmanian with an Adams operation applied.

        Args:
        -----
        i : int
            The degree of the Adams operator.
        as_symbol : bool, optional
            If True, returns the Adams variable as a SymPy Symbol.

        Returns:
        --------
        sp.Expr
            The Grassmanian with the Adams operator applied.
        """
        return sp.Add(*[self.lef ** (i*exp) for exp in self.exponents])
  
    def get_lambda_var(self, i: int, as_symbol: bool = False) -> sp.Expr:
        """
        Returns the Grassmanian with a Lambda operation applied to it.

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
            The Grassmanian with the Lambda operator applied.
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
    
    @staticmethod
    def _get_partitions(height: int, width: int) -> list[list[int]]:
        """
        Returns the set of partitions of the Young diagram, which consist of finding
        sequences of values to to fit in a fixed size box subject to certain constraints.

        Args:
        ----- 
        height : int
            Height of the box.
        width: int
            Width of the box

        Returns:
        --------
        sp.Expr
            The Grassmanian with the Adams operator applied.
        """
        if height == 1:
            return [[i] for i in range(width+1)]
        partitions = []
        for l in range(width + 1):
            for next_partition in Grass._get_partitions(height-1, l):
                partitions.append([l] + next_partition)
        return partitions
    
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
    