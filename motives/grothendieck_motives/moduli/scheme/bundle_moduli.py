from ...motive import Motive
from ...curves.curve import Curve

from ...lefschetz import Lefschetz

class BundleModuli():
    def __init__(self, x: Curve):
        """
        Abstract class describing a moduli space of decorated bundles on a smooth
        complex projective curve.

        Args:
        -----
        x : Curve
            The curve motive on which the moduli depends.
        """
        self.cur = x
        self.g = self.cur.g
        self.lef = Lefschetz()