import sympy as sp

from motives.free import Free


class VectorBundle(Free):
    def __new__(
        cls,
        name: str,
        scheme=None,
        rank=None,
        chern_classes=None,
        chern_character=None,
        max_chern_degree: int = 5,
    ):
        obj = super().__new__(cls, name)

        obj.scheme = scheme
        obj.rank = rank

        degree = rank if rank is not None else max_chern_degree

        if chern_classes is None:
            obj.chern_classes = tuple(
                sp.Symbol(f"c{i}_{name}") for i in range(degree + 1)
            )
        else:
            obj.chern_classes = tuple(map(sp.sympify, chern_classes))

        if chern_character is None:
            obj.chern_character = tuple(
                sp.Symbol(f"ch{i}_{name}") for i in range(degree + 1)
            )
        else:
            obj.chern_character = tuple(map(sp.sympify, chern_character))

        return obj
    
    def set_chern_classes(self, chern_classes):
        """
        Set the Chern classes of the vector bundle.

        Example:
            E.set_chern_classes([1, c1, c2])
        """
        chern_classes = tuple(map(sp.sympify, chern_classes))

        if self.rank is not None and len(chern_classes) != self.rank + 1:
            raise ValueError(
                f"Expected {self.rank + 1} Chern classes for rank {self.rank}, "
                f"but got {len(chern_classes)}."
            )

        if self.rank is None:
            self.rank = len(chern_classes) - 1

        self.chern_classes = chern_classes
        return self

    def set_chern_character(self, chern_character):
        """
        Set the Chern character components of the vector bundle.

        Example:
            E.set_chern_character([2, ch1, ch2])
        """
        self.chern_character = tuple(map(sp.sympify, chern_character))
        return self

    def set_chern_data(self, chern_classes=None, chern_character=None):
        """
        Set both Chern classes and Chern character.

        Example:
            E.set_chern_data(
                chern_classes=[1, c1, c2],
                chern_character=[2, ch1, ch2],
            )
        """
        if chern_classes is not None:
            self.set_chern_classes(chern_classes)

        if chern_character is not None:
            self.set_chern_character(chern_character)

        return self

    def c(self, i: int):
        """
        Return the i-th Chern class.
        """
        if self.chern_classes is None:
            raise ValueError(f"No Chern classes have been assigned to {self}.")

        return self.chern_classes[i]

    def ch(self, i: int):
        """
        Return the i-th Chern character component.
        """
        if self.chern_character is None:
            raise ValueError(f"No Chern character has been assigned to {self}.")

        return self.chern_character[i]