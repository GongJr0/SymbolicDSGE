from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

NDF = NDArray[np.float64]

InpSources = Literal[
    "states",
    "observables",
    "x_pred",
    "x_filt",
    "y_pred",
    "y_filt",
    "innov",
    "std_innov",
    "payload",
]
