"""MCMCResult construction for tests that assert on part of the container.

The container holds the sampler's own bookkeeping, phase acceptance counts and
phase timings, next to the posterior it drew. A test usually exercises one side
or the other, and states only the fields it asserts on. Overrides reach the
constructor exactly as given: nothing here coerces a dtype or a shape, so a
conversion a test is checking for stays the library's to make.

The four bookkeeping defaults are mutually distinct, which lets a round-trip
assertion tell a field that survived from one that was rebuilt from a default.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from SymbolicDSGE.estimation.results import MCMCResult

#: Enough draws for the trace methods to have something to sort, and a burn-in
#: and thin that leave the three acceptance rates at distinct values.
_N_DRAWS = 4
_PARAM_NAMES = ("a", "b")


def make_mcmc_result(**overrides: Any) -> MCMCResult:
    """Build an ``MCMCResult``, taking every unstated field from the defaults."""
    param_names = tuple(overrides.pop("param_names", _PARAM_NAMES))
    samples = overrides.pop(
        "samples", np.zeros((_N_DRAWS, len(param_names)), dtype=np.float64)
    )
    n_rows = int(samples.shape[0])

    fields: dict[str, Any] = {
        "param_names": param_names,
        "samples": samples,
        "logpost_trace": np.zeros(n_rows, dtype=np.float64),
        "logjac_trace": np.zeros(n_rows, dtype=np.float64),
        "accepted_burn_in": 1,
        "accepted_in_sample": 3,
        "burn_in_seconds": 0.25,
        "in_sample_seconds": 1.5,
        "n_draws": n_rows,
        "burn_in": 2,
        "thin": 1,
    }
    fields.update(overrides)
    return MCMCResult(**fields)
