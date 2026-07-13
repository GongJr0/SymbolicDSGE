"""Coverage for the small-N Jarque-Bera critical-value lookup and interpolation.

The existing JB tests exercise only the large-N (chi2) path, so the small-N
table interpolation helpers and the ``JarqueBeraDist`` small-N branch are
untested. These drive every branch of the ``_find_hilo_*`` / ``_isf_interp`` /
``_pval_interp`` kernels and the distribution wrappers.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE._diag_tests import jb_lookup as J


# --- _find_hilo_ascending -------------------------------------------------
def test_find_hilo_ascending_branches():
    grid = J.JB_N_GRID
    # below the grid -> (0, 0)
    assert J._find_hilo_ascending(1, grid) == (0, 0)
    # above the grid -> (last, last)
    assert J._find_hilo_ascending(10_000_000, grid) == (
        grid.shape[0] - 1,
        grid.shape[0] - 1,
    )
    # exact grid hit -> (idx, idx)
    hit = int(grid[3])
    assert J._find_hilo_ascending(hit, grid) == (3, 3)
    # strictly between two nodes -> (idx-1, idx)
    mid = int((grid[3] + grid[4]) // 2)
    lo, hi = J._find_hilo_ascending(mid, grid)
    assert lo + 1 == hi


# --- _find_hilo_descending ------------------------------------------------
def test_find_hilo_descending_branches():
    # a strictly descending array (critical values shrink as p grows)
    arr = J.JB_SMALL_N_CRITICAL_VALUES[:, 0]
    assert arr[0] > arr[-1]
    # above the largest value -> (0, 0)
    assert J._find_hilo_descending(np.float64(arr[0] + 10.0), arr) == (0, 0)
    # below the smallest value -> (last, last)
    assert J._find_hilo_descending(np.float64(arr[-1] - 1.0), arr) == (
        arr.shape[0] - 1,
        arr.shape[0] - 1,
    )
    # exact hit
    assert J._find_hilo_descending(np.float64(arr[5]), arr) == (5, 5)
    # between two nodes
    mid = np.float64((arr[5] + arr[6]) / 2.0)
    lo, hi = J._find_hilo_descending(mid, arr)
    assert lo + 1 == hi


# --- _isf_interp ----------------------------------------------------------
def test_isf_interp_edge_values():
    assert np.isnan(J._isf_interp(50, np.float64(np.nan)))
    assert np.isinf(J._isf_interp(50, np.float64(0.0)))
    assert J._isf_interp(50, np.float64(1.0)) == 0.0


def test_isf_interp_grid_and_interior():
    n_node = int(J.JB_N_GRID[4])
    p_node = np.float64(J.JB_PVAL_GRID[5])
    # exact grid node on both axes
    exact = J._isf_interp(n_node, p_node)
    assert exact == pytest.approx(J.JB_SMALL_N_CRITICAL_VALUES[5, 4])
    # n exact, p interpolated
    p_mid = np.float64((J.JB_PVAL_GRID[5] + J.JB_PVAL_GRID[6]) / 2.0)
    v = J._isf_interp(n_node, p_mid)
    lo, hi = J.JB_SMALL_N_CRITICAL_VALUES[5, 4], J.JB_SMALL_N_CRITICAL_VALUES[6, 4]
    assert min(lo, hi) <= v <= max(lo, hi)
    # p exact, n interpolated
    n_mid = int((J.JB_N_GRID[4] + J.JB_N_GRID[5]) // 2)
    v2 = J._isf_interp(n_mid, p_node)
    assert np.isfinite(v2)
    # both interpolated (bilinear branch)
    v3 = J._isf_interp(n_mid, p_mid)
    assert np.isfinite(v3)


# --- _pval_interp ---------------------------------------------------------
def test_pval_interp_edge_values():
    assert np.isnan(J._pval_interp(50, np.float64(np.nan)))
    assert J._pval_interp(50, np.float64(0.0)) == 1.0
    assert J._pval_interp(50, np.float64(np.inf)) == 0.0


def test_pval_interp_grid_and_interior():
    n_node = int(J.JB_N_GRID[4])
    # exact critical value on an exact n node -> exact p grid value
    cv = np.float64(J.JB_SMALL_N_CRITICAL_VALUES[5, 4])
    assert J._pval_interp(n_node, cv) == pytest.approx(J.JB_PVAL_GRID[5])
    # x between two critical values, exact n
    x_mid = np.float64(
        (J.JB_SMALL_N_CRITICAL_VALUES[5, 4] + J.JB_SMALL_N_CRITICAL_VALUES[6, 4]) / 2.0
    )
    p = J._pval_interp(n_node, x_mid)
    assert 0.0 <= p <= 1.0
    # n interpolated branch
    n_mid = int((J.JB_N_GRID[4] + J.JB_N_GRID[5]) // 2)
    p2 = J._pval_interp(n_mid, x_mid)
    assert 0.0 <= p2 <= 1.0
    # n interpolated, x landing exactly on an interpolated node
    p3 = J._pval_interp(n_mid, np.float64(J.JB_SMALL_N_CRITICAL_VALUES[5, 4]))
    assert 0.0 <= p3 <= 1.0


# --- array kernels --------------------------------------------------------
def test_interp_array_kernels():
    ps = np.ascontiguousarray([0.01, 0.05, 0.5], dtype=np.float64)
    isf = J._isf_interp_array(60, ps)
    assert isf.shape == ps.shape
    assert np.all(np.isfinite(isf))

    xs = np.ascontiguousarray([1.0, 5.0, 20.0], dtype=np.float64)
    pv = J._pval_interp_array(60, xs)
    assert pv.shape == xs.shape
    assert np.all((pv >= 0.0) & (pv <= 1.0))


# --- _as_distribution_output ----------------------------------------------
def test_as_distribution_output_scalar_and_array():
    scalar = J._as_distribution_output(3.5)
    assert isinstance(scalar, np.floating)
    assert scalar == pytest.approx(3.5)
    arr = J._as_distribution_output(np.array([1.0, 2.0]))
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2,)


# --- JarqueBeraDist small-N vs large-N ------------------------------------
def test_jarque_bera_dist_small_n_paths():
    d = J.JarqueBeraDist(n=50)
    assert bool(d._small_n)
    # scalar paths
    sf = d.sf(5.0)
    assert 0.0 <= float(sf) <= 1.0
    cdf = d.cdf(5.0)
    assert float(cdf) == pytest.approx(1.0 - float(sf))
    isf = d.isf(0.05)
    assert np.isfinite(float(isf))
    ppf = d.ppf(0.95)
    assert np.isfinite(float(ppf))
    # array paths (reshape branch in _small_n_sf / _small_n_isf)
    sf_arr = d.sf(np.array([[2.0, 5.0], [10.0, 20.0]]))
    assert sf_arr.shape == (2, 2)
    isf_arr = d.isf(np.array([0.1, 0.5, 0.9]))
    assert isf_arr.shape == (3,)


def test_jarque_bera_dist_large_n_uses_chi2():
    d = J.JarqueBeraDist(n=50_000)
    assert not d._small_n
    # large-N delegates to chi2(df=2); just confirm the wrappers return finite
    assert 0.0 <= float(d.sf(5.0)) <= 1.0
    assert 0.0 <= float(d.cdf(5.0)) <= 1.0
    assert np.isfinite(float(d.isf(0.05)))
    assert np.isfinite(float(d.ppf(0.95)))
