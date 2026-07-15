from motives.core.operator.ring_operator import Sigma, Lambda_


class Wedge(Sigma):
    """
    Exterior power operation.

    Internally, it behaves like Sigma, because in the current motives
    machinery Sigma is the formal operation we want to use for wedge.

    Visually, it prints as ∧.
    """

    def _sympystr(self, printer):
        degree, operand = self.args
        return f"∧{printer.doprint(degree)}({printer.doprint(operand)})"

    def _latex(self, printer):
        degree, operand = self.args
        return (
            r"\wedge^{%s}\left(%s\right)"
            % (printer._print(degree), printer._print(operand))
        )


class SymPower(Lambda_):
    """
    Symmetric power operation.

    Internally, it behaves like Lambda_, because the current motives
    machinery expands Lambda_ in the way we want for symmetric powers.

    Visually, it prints as Sym.
    """

    def _sympystr(self, printer):
        degree, operand = self.args
        return f"Sym{printer.doprint(degree)}({printer.doprint(operand)})"

    def _latex(self, printer):
        degree, operand = self.args
        return (
            r"\operatorname{Sym}^{%s}\left(%s\right)"
            % (printer._print(degree), printer._print(operand))
        )