from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias, cast, overload

import numpy as np
from numpy import float64, int64
from numpy.typing import NDArray
from numba import njit

from scipy.stats import chi2
from scipy.stats._distn_infrastructure import rv_frozen

if TYPE_CHECKING:
    import optype.numpy as onp

NDF: TypeAlias = NDArray[float64]
NDI: TypeAlias = NDArray[int64]
DistributionOutput: TypeAlias = float | float64 | NDF

# Small N Critical value tables
JB_N_GRID: NDI = np.ascontiguousarray(
    np.array(
        [10, 20, 35, 50, 75, 100, 150, 200, 300, 500, 800, 1_000, 1_600, 2_400, 10_000],
        dtype=int64,
    ),
)

JB_PVAL_GRID: NDF = np.ascontiguousarray(
    np.array(
        [
            0.0001,
            0.0005,
            0.0010,
            0.0050,
            0.0100,
            0.0500,
            0.1000,
            0.1500,
            0.2000,
            0.3000,
            0.4000,
            0.5000,
            0.6000,
            0.7000,
            0.8000,
            0.8500,
            0.9000,
            0.9500,
            0.9900,
            0.9950,
            0.9990,
            0.9995,
            0.9999,
        ],
        dtype=np.float64,
    ),
)


# fmt: off
JB_SMALL_N_CRITICAL_VALUES: NDF = np.ascontiguousarray(
    np.array([
        [15.345, 46.996, 66.612, 71.734, 69.910, 68.032, 60.632, 54.736, 47.572, 38.847, 33.247, 31.213, 26.956, 24.249, 19.940],
        [12.444, 31.159, 40.759, 43.256, 41.909, 40.430, 37.229, 34.330, 30.561, 26.270, 23.045, 21.979, 19.760, 18.366, 16.052],
        [10.995, 24.970, 31.969, 33.753, 32.738, 31.840, 29.547, 27.551, 24.830, 21.812, 19.521, 18.736, 17.150, 16.083, 14.397],
        [7.304, 13.471, 16.414, 17.281, 17.305, 16.959, 16.257, 15.638, 14.669, 13.583, 12.726, 12.366, 11.762, 11.384, 10.792],
        [5.7029, 9.7182, 11.736, 12.392, 12.586, 12.491, 12.185, 11.882, 11.3580, 10.778, 10.299, 10.117, 9.8095, 9.6084, 9.3128],
        [2.5247, 3.7954, 4.5929, 4.9757, 5.2777, 5.4300, 5.5984, 5.6758, 5.7732, 5.8551, 5.9103, 5.9242, 5.9569, 5.9671, 5.9857],
        [1.6232, 2.3470, 2.8814, 3.1834, 3.4862, 3.6734, 3.9041, 4.0327, 4.1891, 4.3317, 4.4274, 4.4568, 4.5132, 4.5424, 4.5888],
        [1.2826, 1.8230, 2.2533, 2.5094, 2.7713, 2.9390, 3.1416, 3.2580, 3.4003, 3.5312, 3.6198, 3.6507, 3.7016, 3.7309, 3.7778],
        [1.1236, 1.5623, 1.9162, 2.1278, 2.3463, 2.4865, 2.6558, 2.7559, 2.8764, 2.9882, 3.0645, 3.0909, 3.1360, 3.1611, 3.2036],
        [0.9389, 1.2516, 1.4997, 1.6466, 1.7975, 1.8944, 2.0112, 2.0807, 2.1639, 2.2427, 2.2962, 2.3153, 2.3460, 2.3650, 2.3968],
        [0.8077, 1.0360, 1.2115, 1.3128, 1.4165, 1.4828, 1.5619, 1.6087, 1.6649, 1.7175, 1.7547, 1.7679, 1.7889, 1.8024, 1.8248],
        [0.6950, 0.8574, 0.9771, 1.0447, 1.1126, 1.1563, 1.2076, 1.2385, 1.2752, 1.3101, 1.3338, 1.3420, 1.3568, 1.3655, 1.3808],
        [0.5885, 0.6948, 0.7699, 0.8114, 0.8529, 0.8800, 0.9105, 0.9292, 0.9518, 0.9732, 0.9882, 0.9931, 1.0024, 1.0085, 1.0181],
        [0.4801, 0.5378, 0.5769, 0.5985, 0.6202, 0.6348, 0.6508, 0.6610, 0.6730, 0.6851, 0.6940, 0.6965, 0.7018, 0.7056, 0.7108],
        [0.3618, 0.3777, 0.3896, 0.3969, 0.4046, 0.4105, 0.4168, 0.4213, 0.4267, 0.4325, 0.4368, 0.4376, 0.4402, 0.4421, 0.4451],
        [0.2950, 0.2938, 0.2958, 0.2982, 0.3010, 0.3044, 0.3071, 0.3096, 0.3130, 0.3163, 0.3189, 0.3194, 0.3209, 0.3221, 0.3245],
        [0.2192, 0.2047, 0.2002, 0.1997, 0.1997, 0.2006, 0.2016, 0.2024, 0.2040, 0.2060, 0.2071, 0.2074, 0.2081, 0.2089, 0.2106],
        [0.1272, 0.1084, 0.1022, 0.1005, 0.0995, 0.0996, 0.0992, 0.0995, 0.1000, 0.1005, 0.1010, 0.1012, 0.1013, 0.1019, 0.1024],
        [0.0304, 0.0230, 0.0208, 0.0203, 0.0198, 0.0197, 0.0196, 0.0196, 0.0197, 0.0198, 0.0197, 0.0199, 0.0200, 0.0200, 0.0200],
        [0.0156, 0.0116, 0.0104, 0.0101, 0.0099, 0.0098, 0.0098, 0.0098, 0.0099, 0.0099, 0.0098, 0.0099, 0.0099, 0.0099, 0.0100],
        [0.0032, 0.0023, 0.0021, 0.0020, 0.0020, 0.0019, 0.0019, 0.0019, 0.0020, 0.0019, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020],
        [0.0016, 0.0012, 0.0010, 0.0010, 0.0010, 0.0009, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010],
        [0.0003, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002],
    ], dtype=np.float64)
)
# fmt: on
@njit(cache=True)
def _find_hilo_ascending(val: int | float64, arr: np.ndarray) -> tuple[int, int]:
    idx = np.searchsorted(arr, val)

    length = arr.shape[0]

    if idx == 0:
        return 0, 0
    elif idx == length:
        return length - 1, length - 1
    elif arr[idx] == val:
        return int(idx), int(idx)
    else:
        return int(idx - 1), int(idx)


@njit(cache=True)
def _find_hilo_descending(val: float64, arr: NDF) -> tuple[int, int]:
    idx = np.searchsorted(-arr, -val)

    length = arr.shape[0]

    if idx == 0:
        return 0, 0
    elif idx == length:
        return length - 1, length - 1
    elif arr[idx] == val:
        return int(idx), int(idx)
    else:
        return int(idx - 1), int(idx)


@njit(cache=True)
def _isf_interp(n: int, p: float64) -> float64:
    if np.isnan(p):
        return float64(np.nan)
    if p <= 0.0:
        return float64(np.inf)
    if p >= 1.0:
        return float64(0.0)

    n_lo, n_hi = _find_hilo_ascending(n, JB_N_GRID)
    p_lo, p_hi = _find_hilo_ascending(p, JB_PVAL_GRID)

    if n_lo == n_hi and p_lo == p_hi:
        return float64(JB_SMALL_N_CRITICAL_VALUES[p_lo, n_lo])

    elif n_lo == n_hi:
        w_p = (p - JB_PVAL_GRID[p_lo]) / (JB_PVAL_GRID[p_hi] - JB_PVAL_GRID[p_lo])
        return float64(
            (1 - w_p) * JB_SMALL_N_CRITICAL_VALUES[p_lo, n_lo]
            + w_p * JB_SMALL_N_CRITICAL_VALUES[p_hi, n_lo]
        )

    elif p_lo == p_hi:
        w_n = (n - JB_N_GRID[n_lo]) / (JB_N_GRID[n_hi] - JB_N_GRID[n_lo])
        return float64(
            (1 - w_n) * JB_SMALL_N_CRITICAL_VALUES[p_lo, n_lo]
            + w_n * JB_SMALL_N_CRITICAL_VALUES[p_lo, n_hi]
        )

    else:
        w_n = (n - JB_N_GRID[n_lo]) / (JB_N_GRID[n_hi] - JB_N_GRID[n_lo])
        w_p = (p - JB_PVAL_GRID[p_lo]) / (JB_PVAL_GRID[p_hi] - JB_PVAL_GRID[p_lo])

        val_lo = (1 - w_n) * JB_SMALL_N_CRITICAL_VALUES[
            p_lo, n_lo
        ] + w_n * JB_SMALL_N_CRITICAL_VALUES[p_lo, n_hi]
        val_hi = (1 - w_n) * JB_SMALL_N_CRITICAL_VALUES[
            p_hi, n_lo
        ] + w_n * JB_SMALL_N_CRITICAL_VALUES[p_hi, n_hi]

        return float64((1 - w_p) * val_lo + w_p * val_hi)


@njit(cache=True)
def _pval_interp(n: int, x: float64) -> float64:
    if np.isnan(x):
        return float64(np.nan)
    if x <= 0.0:
        return float64(1.0)
    if np.isinf(x):
        return float64(0.0)

    n_lo, n_hi = _find_hilo_ascending(n, JB_N_GRID)

    lo_cv = JB_SMALL_N_CRITICAL_VALUES[:, n_lo]
    hi_cv = JB_SMALL_N_CRITICAL_VALUES[:, n_hi]

    if n_lo == n_hi:
        p_lo, p_hi = _find_hilo_descending(x, lo_cv)

        if p_lo == p_hi:
            return float64(JB_PVAL_GRID[p_lo])
        else:
            w_p = (x - lo_cv[p_lo]) / (lo_cv[p_hi] - lo_cv[p_lo])
            return float64((1 - w_p) * JB_PVAL_GRID[p_lo] + w_p * JB_PVAL_GRID[p_hi])
    else:
        w_n = (n - JB_N_GRID[n_lo]) / (JB_N_GRID[n_hi] - JB_N_GRID[n_lo])
        cv = (1 - w_n) * lo_cv + w_n * hi_cv

        p_lo, p_hi = _find_hilo_descending(x, cv)

        if p_lo == p_hi:
            return float64(JB_PVAL_GRID[p_lo])
        else:
            w_p = (x - cv[p_lo]) / (cv[p_hi] - cv[p_lo])
            return float64((1 - w_p) * JB_PVAL_GRID[p_lo] + w_p * JB_PVAL_GRID[p_hi])


@njit(cache=True)
def _isf_interp_array(n: int, p: NDF) -> NDF:
    out = np.empty_like(p)
    for i in range(p.size):
        out[i] = _isf_interp(n, p[i])
    return out


@njit(cache=True)
def _pval_interp_array(n: int, x: NDF) -> NDF:
    out = np.empty_like(x)
    for i in range(x.size):
        out[i] = _pval_interp(n, x[i])
    return out


def _as_distribution_output(value: Any) -> DistributionOutput:
    values = np.asarray(value, dtype=np.float64)
    if values.ndim == 0:
        return float64(values.item())
    return values


class JarqueBeraDist(rv_frozen):
    def __init__(self, n: int | np.integer[Any]):
        super().__init__(chi2, df=2)
        self.n = int(n)
        self._small_n = self.n <= JB_N_GRID[-1]

    @overload
    def cdf(self, x: onp.ToFloat, /) -> DistributionOutput: ...

    @overload
    def cdf(self, x: onp.ToFloatND, /) -> NDF: ...

    def cdf(self, x: Any, /) -> DistributionOutput:
        if self._small_n:
            return 1.0 - self._small_n_sf(x)
        else:
            return _as_distribution_output(super().cdf(x))

    @overload
    def sf(self, x: onp.ToFloat, /) -> DistributionOutput: ...

    @overload
    def sf(self, x: onp.ToFloatND, /) -> NDF: ...

    def sf(self, x: Any, /) -> DistributionOutput:
        if self._small_n:
            return self._small_n_sf(x)
        else:
            return _as_distribution_output(super().sf(x))

    @overload
    def ppf(self, q: onp.ToFloat, /) -> DistributionOutput: ...

    @overload
    def ppf(self, q: onp.ToFloatND, /) -> NDF: ...

    def ppf(self, q: Any, /) -> DistributionOutput:
        if self._small_n:
            return self._small_n_isf(1.0 - np.asarray(q, dtype=np.float64))
        else:
            return _as_distribution_output(super().ppf(q))

    @overload
    def isf(self, q: onp.ToFloat, /) -> DistributionOutput: ...

    @overload
    def isf(self, q: onp.ToFloatND, /) -> NDF: ...

    def isf(self, q: Any, /) -> DistributionOutput:
        if self._small_n:
            return self._small_n_isf(q)
        else:
            return _as_distribution_output(super().isf(q))

    def _small_n_sf(self, x: Any, /) -> DistributionOutput:
        values = np.asarray(x, dtype=np.float64)
        if values.ndim == 0:
            return float64(_pval_interp(self.n, float64(values.item())))
        flat = np.ascontiguousarray(values.reshape(-1))
        return cast(NDF, _pval_interp_array(self.n, flat).reshape(values.shape))

    def _small_n_isf(self, q: Any, /) -> DistributionOutput:
        values = np.asarray(q, dtype=np.float64)
        if values.ndim == 0:
            return float64(_isf_interp(self.n, float64(values.item())))
        flat = np.ascontiguousarray(values.reshape(-1))
        return cast(NDF, _isf_interp_array(self.n, flat).reshape(values.shape))
