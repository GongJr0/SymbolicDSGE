from typing import NamedTuple

class ArenaOffset(NamedTuple):
    foffset: tuple[int, ...]
    fwidth: tuple[int, ...]
    ioffset: tuple[int, ...]
    iwidth: tuple[int, ...]

def transform_offsets(kind: str, n: int, p: int, param: int = 0) -> ArenaOffset: ...
def regression_offsets(
    kind: str, n: int, p: int, intercept: bool, n_alpha: int = 0, max_iter: int = 0
) -> ArenaOffset: ...
def regression_output_offsets(kind: str, p: int) -> ArenaOffset: ...
def simulation_offsets(
    order: int, n_state: int, n_var: int, n_exog: int, T: int, n_par: int
) -> ArenaOffset: ...
def simulation_output_offsets(
    order: int, n_var: int, n_exog: int, T: int, n_obs: int
) -> ArenaOffset: ...
def raw_model_data_output_offsets(
    n_states: int, n_shocks: int, n_observables: int
) -> ArenaOffset: ...
def filter_offsets(
    kind: str, n_state: int, n_ctrl: int, n_exog: int, n_obs: int, T: int, n_par: int
) -> ArenaOffset: ...
def filter_output_offsets(
    kind: str,
    n_state: int,
    n_ctrl: int,
    n_exog: int,
    n_obs: int,
    T: int,
    return_shocks: bool = False,
) -> ArenaOffset: ...
def diagnostic_offsets(kind: str, n: int, p: int = 0, lags: int = 0) -> ArenaOffset: ...
def diagnostic_output_offsets() -> ArenaOffset: ...
