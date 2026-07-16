import sympy as sp

from motives.free import Free


class VectorBundle(Free):
    """
    Symbolic vector bundle over a scheme.

    A ``VectorBundle`` is a SymPy-compatible atomic expression together with
    geometric metadata and characteristic data. It may therefore appear
    inside symbolic sums, products, powers, exterior powers, and symmetric
    powers.

    The stored characteristic data are indexed by degree:

    ``c(E) = (c_0(E), c_1(E), ..., c_N(E))``

    and

    ``ch(E) = (ch_0(E), ch_1(E), ..., ch_N(E))``.

    The degree-zero conventions are

    ``c_0(E) = 1``

    and

    ``ch_0(E) = rank(E)``.

    Parameters
    ----------
    name : str
        Symbolic name of the bundle. It is also used to generate default
        symbols for its rank and characteristic classes.
    scheme : Scheme
        Base scheme over which the bundle is defined.
    rank : int or sympy.Expr, optional
        Rank of the bundle. If omitted, the symbol ``rk_<name>`` is created.
    max_degree : int, default=5
        Fallback maximum characteristic degree when the scheme dimension is
        not an integer.
    chern_classes : None, list, tuple, or dict, optional
        Initial Chern classes.

        - ``None`` creates ``(1, c1_<name>, ..., cN_<name>)``.
        - A list or tuple replaces all stored components and is padded with
          zeros when necessary.
        - A dictionary updates selected components by degree.
    chern_character : None, list, tuple, or dict, optional
        Initial homogeneous Chern-character components.

        - ``None`` creates ``(rank, ch1_<name>, ..., chN_<name>)``.
        - A list or tuple replaces all stored components and is padded with
          zeros when necessary.
        - A dictionary updates selected components by degree.
    **assumptions
        SymPy assumptions passed to the underlying symbolic object.

    Attributes
    ----------
    scheme : Scheme
        Base scheme of the bundle.
    rank : sympy.Expr
        Rank of the bundle.
    max_degree : int
        Highest stored characteristic degree.
    chern_classes : tuple of sympy.Expr
        Chern classes indexed by degree.
    chern_character : tuple of sympy.Expr
        Chern-character components indexed by degree.

    Notes
    -----
    When ``scheme.dimension`` is an integer ``d``, characteristic data are
    stored through degree ``d``. Otherwise, ``max_degree`` is used.

    The symbolic identity created by SymPy is determined by ``name`` and the
    SymPy assumptions. The scheme, rank, and characteristic data are metadata
    initialized after construction and are not part of the underlying SymPy
    arguments.
    """

    def __new__(cls, name: str, scheme, rank=None, max_degree: int = 5, chern_classes=None, chern_character=None, **assumptions):
        """
        Create the atomic SymPy expression representing the bundle.

        Only the symbolic name and SymPy assumptions are used to construct the
        underlying ``Free`` object. Bundle metadata, including the scheme, rank,
        and characteristic classes, is initialized later by ``__init__``.

        Parameters
        ----------
        name : str
            Symbolic name of the bundle.
        scheme
            Base scheme. This parameter is accepted for constructor consistency
            but is not included in the underlying SymPy arguments.
        rank, max_degree, chern_classes, chern_character
            Metadata parameters initialized by ``__init__``.
        **assumptions
            SymPy assumptions associated with the symbolic bundle.

        Returns
        -------
        VectorBundle
            Atomic symbolic vector-bundle expression.
        """
        return Free.__new__(cls, name, **assumptions)


    def __init__(self, name: str, scheme, rank=None, max_degree: int = 5, chern_classes=None, chern_character=None, **assumptions):
        """
        Initialize the geometric and characteristic data of the bundle.

        The maximum stored degree is taken from ``scheme.dimension`` when that
        value is an integer. Otherwise, ``max_degree`` is used.

        Parameters
        ----------
        name : str
            Name of the bundle.
        scheme : Scheme
            Base scheme of the bundle.
        rank : int or sympy.Expr, optional
            Bundle rank. Defaults to ``rk_<name>``.
        max_degree : int, default=5
            Fallback truncation degree.
        chern_classes : None, list, tuple, or dict, optional
            Initial Chern-class data.
        chern_character : None, list, tuple, or dict, optional
            Initial Chern-character data.
        **assumptions
            SymPy assumptions already handled during ``__new__``.
        """
        self.name = name
        self.scheme = scheme

        self.rank = (
            sp.sympify(rank)
            if rank is not None
            else sp.Symbol(f"rk_{name}")
        )

        scheme_dimension = getattr(scheme, "dimension", None)

        if isinstance(scheme_dimension, (int, sp.Integer)):
            self.max_degree = int(scheme_dimension)
        else:
            self.max_degree = max_degree

        self._chern_classes = None
        self._chern_character = None

        self.chern_classes = chern_classes
        self.chern_character = chern_character


    def _default_chern_classes(self):
        """
        Construct the default tuple of Chern classes.

        Returns
        -------
        tuple of sympy.Expr
            Tuple

            ``(1, c1_<name>, c2_<name>, ..., cN_<name>)``,

            where ``N = self.max_degree``.

        Notes
        -----
        The degree-zero component is fixed by the standard normalization

        ``c_0(E) = 1``.
        """
        return tuple(
            [sp.Integer(1)]
            + [
                sp.Symbol(f"c{i}_{self.name}")
                for i in range(1, self.max_degree + 1)
            ]
        )


    def _default_chern_character(self):
        """
        Construct the default homogeneous Chern-character components.

        Returns
        -------
        tuple of sympy.Expr
            Tuple

            ``(rank(E), ch1_<name>, ch2_<name>, ..., chN_<name>)``,

            where ``N = self.max_degree``.

        Notes
        -----
        The degree-zero component satisfies

        ``ch_0(E) = rank(E)``.
        """
        return tuple(
            [self.rank]
            + [
                sp.Symbol(f"ch{i}_{self.name}")
                for i in range(1, self.max_degree + 1)
            ]
        )


    def _update_tuple_attribute(self, current, value, default):
        """
        Initialize, replace, or partially update tuple-valued characteristic data.

        Parameters
        ----------
        current : tuple or None
            Currently stored components.
        value : None, list, tuple, or dict
            Requested update.

            - ``None`` returns the default tuple.
            - A dictionary replaces only the specified indices.
            - A list or tuple replaces the entire tuple and is padded on the right
            with zeros.
        default : tuple
            Default components used when ``value`` is ``None`` or when a partial
            update is applied before any data have been stored.

        Returns
        -------
        tuple of sympy.Expr
            Updated tuple with exactly ``self.max_degree + 1`` components.

        Raises
        ------
        IndexError
            If a dictionary contains an index outside
            ``0 <= index <= self.max_degree``.
        ValueError
            If a list or tuple contains more than
            ``self.max_degree + 1`` components.
        TypeError
            If ``value`` is not ``None``, a dictionary, a list, or a tuple.

        Notes
        -----
        Explicit input is allowed to replace the degree-zero component. Therefore,
        this helper does not independently enforce ``c_0 = 1`` or
        ``ch_0 = rank`` after a user-provided update.
        """
        if value is None:
            return default

        if isinstance(value, dict):
            updated = list(current if current is not None else default)

            for index, new_value in value.items():
                index = int(index)

                if index < 0 or index > self.max_degree:
                    raise IndexError(
                        f"Index {index} is out of range. "
                        f"Expected an index between 0 and {self.max_degree}."
                    )

                updated[index] = sp.sympify(new_value)

            return tuple(updated)

        if isinstance(value, (list, tuple)):
            if len(value) > self.max_degree + 1:
                raise ValueError(
                    f"Expected at most {self.max_degree + 1} components, "
                    f"but got {len(value)}."
                )

            padded = list(value) + [sp.Integer(0)] * (
                self.max_degree + 1 - len(value)
            )

            return tuple(sp.sympify(v) for v in padded)

        raise TypeError(
            "Expected None, a list, a tuple, or a dictionary."
        )


    @property
    def chern_classes(self):
        """
        Return the stored Chern classes.

        Returns
        -------
        tuple of sympy.Expr
            Tuple indexed by degree, where element ``i`` is ``c_i(E)``.
        """
        return self._chern_classes


    @chern_classes.setter
    def chern_classes(self, value):
        """
        Set or update the stored Chern classes.

        Parameters
        ----------
        value : None, list, tuple, or dict
            ``None`` restores the default symbolic classes, a list or tuple
            replaces all components, and a dictionary updates selected degrees.

        Raises
        ------
        IndexError
            If a dictionary index is outside the stored degree range.
        ValueError
            If too many components are supplied.
        TypeError
            If the input has an unsupported type.
        """
        default = self._default_chern_classes()

        self._chern_classes = self._update_tuple_attribute(
            current=self._chern_classes,
            value=value,
            default=default,
        )


    @property
    def chern_character(self):
        """
        Return the stored homogeneous Chern-character components.

        Returns
        -------
        tuple of sympy.Expr
            Tuple indexed by degree, where element ``i`` is ``ch_i(E)``.
        """
        return self._chern_character

    @chern_character.setter
    def chern_character(self, value):
        """
        Set or update the stored Chern-character components.

        Parameters
        ----------
        value : None, list, tuple, or dict
            ``None`` restores the default symbolic components, a list or tuple
            replaces all components, and a dictionary updates selected degrees.

        Raises
        ------
        IndexError
            If a dictionary index is outside the stored degree range.
        ValueError
            If too many components are supplied.
        TypeError
            If the input has an unsupported type.
        """


    def _select_tuple_components(self, values, index):
        """
        Select one or more components of a tuple-valued invariant.

        Parameters
        ----------
        values : tuple
            Stored invariant components indexed by degree.
        index : int, list, tuple, or None
            Component selector.

            - ``None`` returns the entire tuple.
            - An integer returns one component.
            - A list or tuple returns the selected components as a tuple.

        Returns
        -------
        sympy.Expr or tuple of sympy.Expr
            Selected component or components.

        Raises
        ------
        IndexError
            If a requested index lies outside the stored tuple.
        """
        if index is None:
            return values

        if isinstance(index, (list, tuple)):
            return tuple(values[int(i)] for i in index)

        return values[int(index)]


    def c(self, index=None):
        """
        Return one or more Chern classes of the bundle.

        Parameters
        ----------
        index : int, list, tuple, or None, optional
            Degree or degrees to select. If omitted, all stored Chern classes are
            returned.

        Returns
        -------
        sympy.Expr or tuple of sympy.Expr
            Selected Chern class or classes.

        Examples
        --------
        Return the complete tuple:

        ``E.c()``

        Return the first Chern class:

        ``E.c(1)``

        Return several components:

        ``E.c([1, 2])``
        """
        return self._select_tuple_components(
            values=self.chern_classes,
            index=index,
        )


    def ch(self, index=None):
        """
        Return one or more homogeneous Chern-character components.

        Parameters
        ----------
        index : int, list, tuple, or None, optional
            Degree or degrees to select. If omitted, all stored Chern-character
            components are returned.

        Returns
        -------
        sympy.Expr or tuple of sympy.Expr
            Selected Chern-character component or components.

        Examples
        --------
        Return the complete tuple:

        ``E.ch()``

        Return the degree-two component:

        ``E.ch(2)``

        Return several components:

        ``E.ch([1, 2])``
        """
        return self._select_tuple_components(
            values=self.chern_character,
            index=index,
        )