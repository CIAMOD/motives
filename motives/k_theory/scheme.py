from dataclasses import dataclass, field


@dataclass
class Scheme:
    name: str
    dimension: int | None = None
    base_field: str | None = None
    betti_numbers: list[int] | None = field(default=None)

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


@dataclass
class Curve(Scheme):
    genus: int | None = None

    def __post_init__(self):
        self.dimension = 1

        if self.betti_numbers is None and self.genus is not None:
            self.betti_numbers = [1, 2 * self.genus, 1]