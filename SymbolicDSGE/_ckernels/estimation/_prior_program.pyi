"""Type stubs for the compiled ``_prior_program`` extension.

The native kernels carry no inspectable type information, so these signatures
exist solely for the LSP / mypy. They must stay in sync with
``_prior_program.pyx`` / ``prior_program.c`` and the numba references in
``SymbolicDSGE.estimation.prior_program``; the parity tests guard the runtime
behavior, not this stub.
"""

from numpy import float64, int64
from numpy.typing import NDArray

_F64 = NDArray[float64]
_I64 = NDArray[int64]

def dist_logpdf(code: int, params: _F64, x: float) -> float: ...
def transform_inverse_and_logjac(
    code: int, params: _F64, z: float
) -> tuple[float, float]: ...
def lkj_chol_logjac(z: _F64, dim: int, length: int) -> float: ...
def lkj_chol_logpdf_from_z(
    z: _F64, dim: int, length: int, eta: float, log_const: float
) -> float: ...
def logprior_program(
    theta: _F64,
    dist_codes: _I64,
    transform_codes: _I64,
    dist_params: _F64,
    transform_params: _F64,
    include_logjac: bool = True,
) -> float: ...
def cov_from_unconstrained(z: _F64, std: _F64) -> tuple[_F64, _F64]: ...
def unconstrained_from_corr_chol(L: _F64) -> _F64: ...
