from ..core.operand.operand import Operand
import sympy as sp


class Motive(Operand):
    """
    An abstract operand node in an expression tree that represents a motive
    in the Grothendieck lambda-ring of varieties, the Grothendieck rin of
    Chow motives or in any extension or completion of such rings begin considered.
    """

    def Sym(self, n: int) -> sp.Expr:
        """
        Computes the n-th symmetric product of the motive.

        Returns:
        --------
        sp.Expr
            Expression representing the n-th symmetric product of the motive.
        """
        return self.lambda_(n)

    def Alt(self, n: int) -> sp.Expr:
        """
        Computes the n-th alternating product of the motive.

        Returns:
        --------
        sp.Expr
            Expression representing the n-th symmetric product of the motive.
        """
        return self.sigma(n)
