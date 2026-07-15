import sympy as sp


class Scheme:
    """
    Symbolic scheme with dimension and Betti numbers.

    Parameters
    ----------
    name : str
        Name of the scheme. It is used for printing and for generating
        default symbolic Betti numbers.

    dimension
        Dimension of the scheme. If it is an integer, the maximum Betti degree
        is set to ``2 * dimension``.

    betti_numbers
        Optional tuple or list of Betti numbers. If not provided, symbolic
        Betti numbers ``b0_<name>, ..., bN_<name>`` are created.

    max_betti_degree : int
        Maximum Betti degree used when the dimension is not available as an
        integer.
    """

    def __init__(
        self,
        name: str,
        dimension,
        betti_numbers=None,
        max_betti_degree: int = 10,
    ):
        """Initialize the scheme metadata and Betti numbers."""
        self.name = name
        self.dimension = sp.sympify(dimension)

        if isinstance(self.dimension, (int, sp.Integer)):
            self.max_betti_degree = 2 * int(self.dimension)
        else:
            self.max_betti_degree = max_betti_degree

        if betti_numbers is None:
            self.betti_numbers = tuple(
                sp.Symbol(f"b{i}_{name}")
                for i in range(self.max_betti_degree + 1)
            )
        else:
            self.betti_numbers = tuple(sp.sympify(b) for b in betti_numbers)

    def __repr__(self) -> str:
        """Return the scheme name as its representation."""
        return self.name

    def __str__(self) -> str:
        """Return the scheme name as a string."""
        return self.name


class Curve(Scheme):
    """
    Symbolic smooth projective curve.

    Parameters
    ----------
    name : str
        Name of the curve.

    genus
        Genus of the curve. If not provided, the symbolic genus ``g`` is used.

    betti_numbers
        Optional tuple or list of Betti numbers. If not provided, the default
        Betti numbers are ``(1, 2g, 1)``.
    """

    def __init__(
        self,
        name: str,
        genus=None,
        betti_numbers=None,
    ):
        """Initialize a curve as a one-dimensional scheme."""
        self.genus = sp.sympify(genus) if genus is not None else sp.Symbol("g")

        if betti_numbers is None:
            betti_numbers = (
                sp.Integer(1),
                2 * self.genus,
                sp.Integer(1),
            )

        super().__init__(
            name=name,
            dimension=1,
            betti_numbers=betti_numbers,
        )