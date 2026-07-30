"""Per-replication Monte Carlo statistical tests.

Shared contract for every ``*_test_step`` factory in this group:

- name: unique step name; also the trace-key prefix (see output below).
- Input selectors: pass ``source`` and ``field`` strings.
- Select/trim: ``column``/``columns`` to pick columns, ``burn_in`` to drop leading
  rows, ``drop_initial`` to drop the initial x0 row.
- Shared kwargs: ``alpha=0.05``.
- Output location: ``traces["test.<name>.statistic" | ".pval" | ".status"]`` (length n_rep).
"""

from .builtins import (
    breusch_godfrey_test_step,
    breusch_pagan_test_step,
    chow_test_step,
    cusum_test_step,
    cusumsq_test_step,
    jarque_bera_test_step,
    ljung_box_test_step,
    wald_test_step,
)

__all__ = [
    "wald_test_step",
    "ljung_box_test_step",
    "jarque_bera_test_step",
    "breusch_pagan_test_step",
    "breusch_godfrey_test_step",
    "cusum_test_step",
    "cusumsq_test_step",
    "chow_test_step",
]
