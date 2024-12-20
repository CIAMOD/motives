# -------- BASE ----------
from .free import Free
from .integer import Integer

# -------- CORE ----------
from .core.lambda_ring_context import LambdaRingContext
from .core.lambda_ring_expr import LambdaRingExpr
from .core.operand import Operand
from .core.object_1_dim import Object1Dim

# -------- OPERATOR ----------
from .core.operator import Lambda_, Sigma, Adams

# -------- GROTHENDIECK ----------
from .grothendieck_motives.motive import Motive
from .grothendieck_motives.point import Point
from .grothendieck_motives.polynomial_1_var import Polynomial1Var
from .grothendieck_motives.proj import Proj
from .grothendieck_motives.lefschetz import Lefschetz

## -------- CURVES ----------
from .grothendieck_motives.curves.curve import Curve
from .grothendieck_motives.curves.curvehodge import CurveHodge
from .grothendieck_motives.curves.jacobian import Jacobian

## -------- GL ----------
from .grothendieck_motives.groups.gl import GL
from .grothendieck_motives.groups.g import G
from .grothendieck_motives.groups.gl import GL
from .grothendieck_motives.groups.o import O
from .grothendieck_motives.groups.pgl import PGL
from .grothendieck_motives.groups.pso import PSO
from .grothendieck_motives.groups.sl import SL
from .grothendieck_motives.groups.so import SO
from .grothendieck_motives.groups.sp import Sp

## -------- MODULI ----------
### -------- SCHEME ----------
from .grothendieck_motives.moduli.scheme.twisted_higgs_moduli import TwistedHiggsModuli
from .grothendieck_motives.moduli.scheme.twisted_higgs_moduli_bb import (
    TwistedHiggsModuliBB,
)
from .grothendieck_motives.moduli.scheme.twisted_higgs_moduli_adhm import (
    TwistedHiggsModuliADHM,
)
from .grothendieck_motives.moduli.scheme.vhs import VHS
from .grothendieck_motives.moduli.scheme.vector_bundle_moduli import VectorBundleModuli

### -------- STACK ----------
from .grothendieck_motives.moduli.stack.bg import BG
from .grothendieck_motives.moduli.stack.bun import Bun
