"""Every arena sizer's output, pinned against a recorded snapshot.

The sizers are being rewritten to walk their own offsets rather than state a
closed form, so what matters is that the numbers do not move. This sweeps each
one over its parameters and compares against ``arena_sizes.json``, which was
recorded from the closed forms.

Regenerate deliberately, never to make a failure go away:

    python -m tests.ckernels.test_arena_size_snapshot
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from SymbolicDSGE._ckernels.monte_carlo import _arena

SNAPSHOT = Path(__file__).with_name("arena_sizes.json")

_TRANSFORMS = (
    "passthrough",
    "standardize",
    "log",
    "log_diff",
    "diff",
    "rolling_mean",
    "rolling_var",
    "rolling_std",
)
_REGRESSIONS = (
    "ols",
    "ridge",
    "ridge_gs",
    "lasso",
    "lasso_gs",
    "elastic_net",
    "elastic_net_gs",
)
_FILTERS = ("linear", "extended", "unscented")
_DIAGNOSTICS = (
    "wald_mean",
    "wald_covariance",
    "wald_second_moment",
    "ljung_box",
    "jarque_bera",
    "breusch_pagan",
    "breusch_godfrey",
    "chow",
    "cusum",
    "cusumsq",
)


def _sweep() -> dict[str, list[int]]:
    """Every sizer over a spread of dimensions, keyed by call."""
    out: dict[str, list[int]] = {}

    for kind in _TRANSFORMS:
        for n in (1, 8, 40):
            for p in (1, 3):
                for order in (0, 2):
                    key = f"transform:{kind}:{n}:{p}:{order}"
                    out[key] = list(_arena.transform_arena_size(kind, n, p, order))

    for kind in _REGRESSIONS:
        for n in (4, 50):
            for p in (1, 5):
                for intercept in (False, True):
                    key = f"regression:{kind}:{n}:{p}:{int(intercept)}"
                    out[key] = list(
                        _arena.regression_arena_size(kind, n, p, intercept, 4, 100)
                    )

    for order in (1, 2):
        for n_state in (1, 6):
            for n_var in (2, 20):
                for n_exog in (0, 3):
                    for T in (1, 200):
                        for n_par in (0, 7):
                            if n_var < n_state:
                                continue
                            key = f"sim:{order}:{n_state}:{n_var}:{n_exog}:{T}:{n_par}"
                            out[key] = list(
                                _arena.simulation_arena_size(
                                    order, n_state, n_var, n_exog, T, n_par
                                )
                            )
                            key = f"simout:{order}:{n_var}:{n_exog}:{T}:{n_par}"
                            out[key] = list(
                                _arena.simulation_output_arena_size(
                                    order, n_var, n_exog, T, n_par
                                )
                            )

    for kind in _FILTERS:
        for n_state in (1, 6):
            for n_ctrl in (0, 4):
                for n_exog in (1, 3):
                    for n_obs in (1, 5):
                        for T in (1, 200):
                            key = (
                                f"filter:{kind}:{n_state}:{n_ctrl}:"
                                f"{n_exog}:{n_obs}:{T}"
                            )
                            out[key] = list(
                                _arena.filter_arena_size(
                                    kind, n_state, n_ctrl, n_exog, n_obs, T, 3
                                )
                            )

    for kind in _DIAGNOSTICS:
        for n in (4, 60):
            for p in (1, 4):
                for lags in (0, 5):
                    key = f"diag:{kind}:{n}:{p}:{lags}"
                    out[key] = list(_arena.diagnostic_arena_size(kind, n, p, lags))

    return out


def test_arena_sizes_match_the_snapshot() -> None:
    recorded = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    current = _sweep()

    assert set(current) == set(recorded), "the sweep itself changed shape"
    drifted = {
        key: (recorded[key], current[key])
        for key in recorded
        if recorded[key] != current[key]
    }
    assert (
        not drifted
    ), f"{len(drifted)} sizer(s) moved: {dict(list(drifted.items())[:5])}"


if __name__ == "__main__":
    SNAPSHOT.write_text(json.dumps(_sweep(), indent=1, sort_keys=True), "utf-8")
    print(f"recorded {len(_sweep())} sizer calls to {SNAPSHOT}")
