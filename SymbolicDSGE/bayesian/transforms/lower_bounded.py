from .transform import Transform, OutOfSupportError
from ..support import Support
from typing import overload

import numpy as np
from numpy import float64
from numpy.typing import NDArray


class LowerBoundedTransform(Transform): ...
