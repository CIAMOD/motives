import sympy as sp

from motives.free import Free


class VectorBundle(Free):
    """
    Symbolic vector bundle over a scheme.

    A ``VectorBundle`` is represented as a symbolic object, inheriting from
    ``Free``, together with metadata such as its base scheme, rank, Chern
    classes, and Chern character components.

    The Chern classes and Chern character are stored as tuples indexed by
    degree:

        c(E)  = (c_0(E), c_1(E), ..., c_max_degree(E))
        ch(E) = (ch_0(E), ch_1(E), ..., ch_max_degree(E))

    By convention, ``c_0(E) = 1`` and ``ch_0(E) = rank(E)``.

    Parameters
    ----------
    name : str
        Name of the vector bundle. It is also used to generate default symbolic
        names for the rank, Chern classes, and Chern character components.

    scheme
        Base scheme over which the vector bundle is defined. If the scheme has
        an integer ``dimension`` attribute, that value is used as the maximum
        degree for Chern classes and Chern character components.

    rank
        Rank of the vector bundle. If not provided, a symbolic rank
        ``rk_<name>`` is created.

    max_degree : int
        Fallback maximum degree for Chern classes and Chern character
        components when the scheme dimension is not available as an integer.

    chern_classes
        Optional initial Chern classes. Accepted values are:

        - ``None``: use the default tuple ``(1, c1_name, ..., cN_name)``.
        - ``list`` or ``tuple``: replace the full tuple, padding missing
          components with zeros.
        - ``dict``: update selected components by index.

    chern_character
        Optional initial Chern character components. Accepted values are:

        - ``None``: use the default tuple ``(rank, ch1_name, ..., chN_name)``.
        - ``list`` or ``tuple``: replace the full tuple, padding missing
          components with zeros.
        - ``dict``: update selected components by index.
    """

    def __new__(
        cls,
        name: str,
        scheme,
        rank=None,
        max_degree: int = 5,
        chern_classes=None,
        chern_character=None,
        **assumptions,
    ):
        """
        Create the symbolic SymPy object.

        Metadata such as the scheme, rank, Chern classes, and Chern character
        is initialized later in ``__init__``.
        """
        return Free.__new__(cls, name, **assumptions)


    def __init__(
        self,
        name: str,
        scheme,
        rank=None,
        max_degree: int = 5,
        chern_classes=None,
        chern_character=None,
        **assumptions,
    ):
        """
        Initialize the vector bundle metadata and characteristic data.

        The maximum degree is taken from ``scheme.dimension`` when that value is
        an integer. Otherwise, the provided ``max_degree`` is used.
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
        Build the default tuple of Chern classes.

        The result has the form:

            (1, c1_name, c2_name, ..., c_max_degree_name)

        where the degree zero Chern class is always 1.
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
        Build the default tuple of Chern character components.

        The result has the form:

            (rank, ch1_name, ch2_name, ..., ch_max_degree_name)

        where the degree zero component is the rank of the bundle.
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
        Create or update tuple-valued characteristic data.

        The accepted input formats are:

        - ``None``: return the default tuple.
        - ``dict``: update selected components by index, preserving all other
          current/default components.
        - ``list`` or ``tuple``: replace the full tuple, padding missing
          components with zeros up to ``max_degree``.

        Parameters
        ----------
        current
            Current stored tuple, or ``None`` if no value has been stored yet.

        value
            New value used to initialize, replace, or partially update the
            tuple.

        default
            Default tuple used when ``value`` is ``None`` or when updating a
            tuple that has not been initialized yet.
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
        Tuple of Chern classes of the vector bundle.

        The tuple is indexed by degree, so ``chern_classes[i]`` represents
        ``c_i(E)``.
        """
        return self._chern_classes


    @chern_classes.setter
    def chern_classes(self, value):
        """
        Set or update the Chern classes.

        ``None`` restores defaults, a list/tuple replaces the full tuple, and a
        dictionary updates selected components by index.
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
        Tuple of Chern character components of the vector bundle.

        The tuple is indexed by degree, so ``chern_character[i]`` represents
        ``ch_i(E)``.
        """
        return self._chern_character

    @chern_character.setter
    def chern_character(self, value):
        """
        Set or update the Chern character components.

        ``None`` restores defaults, a list/tuple replaces the full tuple, and a
        dictionary updates selected components by index.
        """
        default = self._default_chern_character()

        self._chern_character = self._update_tuple_attribute(
            current=self._chern_character,
            value=value,
            default=default,
        )


    def _select_tuple_components(self, values, index):
        """
        Select components from a tuple-valued invariant.

        Parameters
        ----------
        values
            Tuple of stored components.

        index
            Component selector. If ``None``, the full tuple is returned. If an
            integer is given, a single component is returned. If a list or tuple
            is given, the selected components are returned as a tuple.
        """
        if index is None:
            return values

        if isinstance(index, (list, tuple)):
            return tuple(values[int(i)] for i in index)

        return values[int(index)]


    def c(self, index=None):
        """
        Return Chern classes.

        Parameters
        ----------
        index
            If ``None``, return the full tuple of Chern classes. If an integer
            is given, return the corresponding component. If a list or tuple is
            given, return the selected components.

        Examples
        --------
        ``E.c()`` returns all Chern classes.

        ``E.c(1)`` returns ``c_1(E)``.

        ``E.c([1, 2])`` returns ``(c_1(E), c_2(E))``.
        """
        return self._select_tuple_components(
            values=self.chern_classes,
            index=index,
        )


    def ch(self, index=None):
        """
        Return Chern character components.

        Parameters
        ----------
        index
            If ``None``, return the full tuple of Chern character components.
            If an integer is given, return the corresponding component. If a
            list or tuple is given, return the selected components.

        Examples
        --------
        ``E.ch()`` returns all Chern character components.

        ``E.ch(1)`` returns ``ch_1(E)``.

        ``E.ch([1, 2])`` returns ``(ch_1(E), ch_2(E))``.
        """
        return self._select_tuple_components(
            values=self.chern_character,
            index=index,
        )