import sympy as sp


class Scheme:
    """
    Symbolic scheme with dimension and Betti-number metadata.

    A ``Scheme`` acts as the base space of symbolic vector bundles. Its
    dimension determines the default truncation degree used for characteristic
    classes and, when the dimension is an integer, the range of Betti numbers.

    Parameters
    ----------
    name : str
        Name used to display the scheme and to construct default symbolic
        Betti numbers.
    dimension : int or sympy.Expr
        Dimension of the scheme. If it is an integer ``d``, Betti numbers are
        stored through cohomological degree ``2d``. A symbolic expression may
        also be used.
    betti_numbers : iterable, optional
        Explicit Betti numbers ordered by cohomological degree. If omitted,
        symbols

        ``b0_<name>, b1_<name>, ..., bN_<name>``

        are generated automatically.
    max_betti_degree : int, default=10
        Maximum degree used for default symbolic Betti numbers when
        ``dimension`` is not an integer.

    Attributes
    ----------
    name : str
        Name of the scheme.
    dimension : sympy.Expr
        SymPy representation of the scheme dimension.
    max_betti_degree : int
        Highest stored Betti-number degree.
    betti_numbers : tuple of sympy.Expr
        Betti numbers indexed by cohomological degree.

    Notes
    -----
    When ``dimension = d`` is an integer, the maximum Betti degree is

    ``max_betti_degree = 2d``.

    This follows the standard cohomological grading of a complex
    ``d``-dimensional variety.
    """

    def __init__(self, name: str, dimension, betti_numbers=None, max_betti_degree: int = 10):
        """
        Initialize the scheme and its Betti-number data.

        If ``dimension`` is an integer ``d``, ``max_betti_degree`` is replaced by
        ``2d``. Otherwise, the explicitly supplied fallback value is used.

        Parameters
        ----------
        name : str
            Name of the scheme.
        dimension : int or sympy.Expr
            Dimension of the scheme.
        betti_numbers : iterable, optional
            Explicit Betti numbers. When omitted, symbolic components are
            generated from degree zero through ``max_betti_degree``.
        max_betti_degree : int, default=10
            Fallback maximum Betti degree for schemes of symbolic dimension.
        """
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
    Symbolic smooth connected projective curve.

    A curve is represented as a one-dimensional scheme together with its
    genus and Betti numbers.

    Parameters
    ----------
    name : str
        Name of the curve.
    genus : int or sympy.Expr, optional
        Genus of the curve. If omitted, the symbolic variable ``g`` is used.
    betti_numbers : iterable, optional
        Explicit Betti numbers. If omitted, the standard Betti numbers of a
        smooth connected projective curve are used:

        ``(b_0, b_1, b_2) = (1, 2g, 1)``.

    Attributes
    ----------
    genus : sympy.Expr
        Genus of the curve.

    Notes
    -----
    The default Betti numbers encode

    ``dim H⁰ = 1``,
    ``dim H¹ = 2g``,
    ``dim H² = 1``.
    """

    def __init__(self, name: str, genus=None, betti_numbers=None):
        """
        Initialize a symbolic curve.

        The dimension is fixed to one. When no explicit Betti numbers are
        supplied, the tuple ``(1, 2g, 1)`` is used.

        Parameters
        ----------
        name : str
            Name of the curve.
        genus : int or sympy.Expr, optional
            Genus of the curve. Defaults to the symbolic variable ``g``.
        betti_numbers : iterable, optional
            Explicit Betti numbers ordered as ``(b_0, b_1, b_2)``.
        """
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