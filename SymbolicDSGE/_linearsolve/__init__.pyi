from typing import Any

from numpy import complex128, float64
from numpy.typing import NDArray

NDF = NDArray[float64]
NDC = NDArray[complex128]

class model:
    f: NDF | NDC
    p: NDF | NDC
    stab: int
    eig: NDC

    def __init__(
        self,
        equations: Any = ...,
        variables: Any = ...,
        costates: Any = ...,
        states: Any = ...,
        exo_states: Any = ...,
        endo_states: Any = ...,
        shock_names: Any = ...,
        parameters: Any = ...,
        shock_prefix: Any = ...,
        n_states: int | None = ...,
        n_exo_states: int | None = ...,
    ) -> None: ...
    def set_ss(self, steady_state: Any) -> None: ...
    def approximate_and_solve(
        self, log_linear: bool = ..., eigenvalue_warnings: bool = ...
    ) -> None: ...

_to_complex: Any
_klein_postprocess: Any

def klein(
    a: NDF,
    b: NDF,
    c: NDF | None,
    _phi: NDF | None,
    n_states: int,
) -> tuple[NDC, NDC, NDC, NDC, int, NDC]: ...
