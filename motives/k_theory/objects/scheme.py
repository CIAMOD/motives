"""Classes representing schemes and their naturally associated vector bundles."""

import sympy as sp

from .vector_bundle import VectorBundle
from ..operations.determinant import Determinant
from ..operations.dual import Dual


class Scheme:
    """Represent a scheme and its main geometric invariants.

    Each scheme is determined by a name, a non-negative integer dimension and
    a collection of Betti numbers. Its dimension must be explicitly known and
    cannot be symbolic.

    If the scheme has complex dimension d, its Betti numbers are stored from
    degree 0 to degree 2d. Therefore, exactly 2d + 1 Betti numbers are required.
    When they are not provided, symbolic values are created automatically.

    The following standard vector bundles are also created:

    - Ox: structure sheaf O_X, a trivial line bundle.
    - Tx: tangent bundle T_X, with rank equal to dim(X).
    - Tx_dual: cotangent bundle T_X^∨ = Dual(T_X).
    - Kx: canonical bundle K_X = det(T_X^∨).

    Since O_X is trivial, its Chern invariants are fixed as

        c(O_X) = (1, 0, ..., 0),
        ch(O_X) = (1, 0, ..., 0).

    Parameters
    ----------
    name : str
        Name used to identify and display the scheme.
    dimension : int
        Non-negative complex dimension of the scheme.
    betti_numbers : list, tuple or None, optional
        Betti numbers from degree 0 to degree 2 * dimension. If omitted,
        symbolic Betti numbers are created.

    Raises
    ------
    TypeError
        If ``dimension`` is not an integer.
    ValueError
        If ``dimension`` is negative or the number of supplied Betti numbers
        is not equal to ``2 * dimension + 1``.
    """

    def __init__(self, name: str, dimension: int, betti_numbers=None):
        """Initialize the scheme, its Betti numbers and its standard bundles."""
        if isinstance(dimension, bool) or not isinstance(dimension, (int, sp.Integer)):
            raise TypeError("dimension must be an integer.")

        if dimension < 0:
            raise ValueError("dimension must be non-negative.")

        self.name = name
        self.dimension = int(dimension)
        self.max_betti_degree = 2 * self.dimension

        if betti_numbers is None:
            self.betti_numbers = tuple(sp.Symbol(f"b{i}_{name}") for i in range(self.max_betti_degree + 1))
        else:
            if len(betti_numbers) != self.max_betti_degree + 1:
                raise ValueError(f"betti_numbers must contain exactly {self.max_betti_degree + 1} components.")

            self.betti_numbers = tuple(sp.sympify(value) for value in betti_numbers)

        self._initialize_standard_bundles()

    def _initialize_standard_bundles(self) -> None:
        """Create the standard vector bundles naturally associated with the scheme.

        The structure sheaf O_X is initialized as a trivial line bundle, so it
        has rank one and all its positive-degree Chern components vanish.

        The tangent bundle T_X has rank equal to the dimension of X. The
        cotangent and canonical bundles are then constructed as

            T_X^∨ = Dual(T_X),
            K_X = det(T_X^∨).
        """
        trivial_invariants = (sp.Integer(1),) + (sp.Integer(0),) * self.dimension

        self.Ox = VectorBundle(
            f"O_{self.name}",
            self,
            rank=1,
            chern_classes=trivial_invariants,
            chern_character=trivial_invariants
        )
        self.Tx = VectorBundle(f"T_{self.name}", self, rank=self.dimension)
        self.Tx_dual = Dual(self.Tx)
        self.Kx = Determinant(self.Tx_dual)

    def __repr__(self) -> str:
        """Return the name of the scheme as its developer representation."""
        return self.name

    def __str__(self) -> str:
        """Return the name of the scheme as its readable representation."""
        return self.name


class Curve(Scheme):
    """Represent a smooth curve and its geometric invariants.

    A curve has complex dimension one. Its default Betti numbers are determined
    by its genus g:

        b0 = 1,
        b1 = 2g,
        b2 = 1.

    The structure sheaf, tangent bundle, cotangent bundle and canonical bundle
    are created by the parent ``Scheme`` class.

    Parameters
    ----------
    name : str
        Name used to identify and display the curve.
    genus : int, sympy.Expr or None, optional
        Genus of the curve. If omitted, a symbolic genus is created.
    betti_numbers : list, tuple or None, optional
        Custom Betti numbers ``(b0, b1, b2)``. If omitted, ``(1, 2g, 1)`` is
        used.

    Raises
    ------
    ValueError
        If custom Betti numbers are provided but do not contain exactly three
        components.
    """

    def __init__(self, name: str, genus=None, betti_numbers=None):
        """Initialize the curve using dimension one and its genus."""
        self.genus = sp.Symbol(f"g_{name}") if genus is None else sp.sympify(genus)

        if betti_numbers is None:
            betti_numbers = (sp.Integer(1), 2 * self.genus, sp.Integer(1))

        super().__init__(name, dimension=1, betti_numbers=betti_numbers)