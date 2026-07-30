from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._arenas import ArenaAllocation
from SymbolicDSGE._diag_tests.distributions import (
    DistributionParameter,
    ReferenceDistribution,
)
from SymbolicDSGE.monte_carlo.native_lowering import FloatInputBinding

NDF = NDArray[np.float64]
NDI = NDArray[np.int64]

class NativeRunResult(NamedTuple):
    status: int
    halt_rep_idx: int
    halt_step_idx: int
    halt_status: int
    wall_elapsed_s: float
    step_elapsed_s_by_worker: NDF | None
    step_counts_by_worker: NDI | None
    step_failures_by_worker: NDI | None

class NativeStep:
    @property
    def name(self) -> str: ...
    @property
    def test_distribution(self) -> ReferenceDistribution | None: ...
    @property
    def test_df(
        self,
    ) -> DistributionParameter | tuple[DistributionParameter, ...] | None: ...

def payload_step(name: str, value: ArrayLike) -> NativeStep: ...
def raw_model_data_step(
    name: str,
    states: ArrayLike | None = None,
    observables: ArrayLike | None = None,
) -> NativeStep: ...
def simulate1_step(
    name: str,
    measurement_addr: int,
    T: int,
    n_var: int,
    n_exog: int,
    n_par: int,
    n_obs: int,
) -> NativeStep: ...
def simulate2_step(
    name: str,
    measurement_addr: int,
    T: int,
    n_state: int,
    n_ctrl: int,
    n_exog: int,
    n_par: int,
    n_obs: int,
) -> NativeStep: ...
def filter_linear_step(
    name: str,
    T: int,
    n_var: int,
    n_obs: int,
    n_exog: int,
    symmetrize: bool = False,
    jitter: float = 0.0,
    return_shocks: bool = False,
) -> NativeStep: ...
def filter_extended_step(
    name: str,
    measurement_addr: int,
    jacobian_addr: int,
    T: int,
    n_var: int,
    n_obs: int,
    n_exog: int,
    n_par: int,
    symmetrize: bool = False,
    jitter: float = 0.0,
    return_shocks: bool = False,
) -> NativeStep: ...
def filter_unscented_step(
    name: str,
    measurement_addr: int,
    T: int,
    n_state: int,
    n_ctrl: int,
    n_exog: int,
    n_obs: int,
    n_par: int,
    alpha: float,
    beta: float,
    kappa: float,
    symmetrize: bool = False,
    jitter: float = 0.0,
) -> NativeStep: ...
def transform_step(
    name: str,
    kind: str,
    n: int,
    p: int,
    ddof: int = 0,
    offset: float = 0.0,
    order: int = 1,
    window: int = 1,
) -> NativeStep: ...
def ols_step(name: str, n: int, p: int, intercept: bool = True) -> NativeStep: ...
def ridge_step(
    name: str, n: int, p: int, alpha: float, intercept: bool = True
) -> NativeStep: ...
def ridge_gs_step(
    name: str,
    n: int,
    p: int,
    start: float,
    stop: float,
    num: int,
    criterion: int,
    intercept: bool = True,
) -> NativeStep: ...
def lasso_step(
    name: str,
    n: int,
    p: int,
    alpha: float,
    max_iter: int = 1000,
    tol: float = 1e-10,
    intercept: bool = True,
) -> NativeStep: ...
def lasso_gs_step(
    name: str,
    n: int,
    p: int,
    start: float,
    stop: float,
    num: int,
    max_iter: int = 1000,
    tol: float = 1e-10,
    intercept: bool = True,
) -> NativeStep: ...
def elastic_net_step(
    name: str,
    n: int,
    p: int,
    alpha: float,
    l1_ratio: float,
    max_iter: int = 1000,
    tol: float = 1e-10,
    intercept: bool = True,
) -> NativeStep: ...
def elastic_net_gs_step(
    name: str,
    n: int,
    p: int,
    start: float,
    stop: float,
    num: int,
    l1_ratio: float,
    criterion: int,
    max_iter: int = 1000,
    tol: float = 1e-10,
    intercept: bool = True,
) -> NativeStep: ...
def wald_step(
    name: str,
    target: ArrayLike,
    n: int,
    q: int,
    manual_bandwidth: int,
    kernel_id: int,
    bandwidth_mode: int,
    kind: int,
) -> NativeStep: ...
def ljung_box_step(name: str, n: int, lags: int) -> NativeStep: ...
def jarque_bera_step(name: str, n: int) -> NativeStep: ...
def breusch_pagan_step(
    name: str, n: int, k: int, robust: bool = False
) -> NativeStep: ...
def breusch_godfrey_step(name: str, n: int, k: int, lags: int) -> NativeStep: ...
def cusum_step(name: str, n: int, p: int) -> NativeStep: ...
def cusumsq_step(name: str, n: int, p: int) -> NativeStep: ...
def chow_step(name: str, n: int, p: int, t_break: int) -> NativeStep: ...
def run(
    allocation: ArenaAllocation,
    steps: Sequence[NativeStep],
    input_bindings: Sequence[Sequence[FloatInputBinding]] | None = None,
    fail_fast: bool = False,
    profile_steps: bool = False,
) -> NativeRunResult: ...
