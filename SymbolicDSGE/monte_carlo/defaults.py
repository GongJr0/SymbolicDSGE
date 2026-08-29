"""Defaults for step kwargs that no native context carries.

A native builder defaults its own scalar knobs, so a node that omits one still
lowers. What remains are the kwargs that never reach a context: selectors the
Python side dispatches on, and values consumed before a step is built. Their
consumers resolve them here.

The step factories restate these in their signatures rather than importing
them, so a caller reads the value it will get.
"""

from __future__ import annotations

DEFAULT_FILTER_MODE = "linear"
DEFAULT_REGRESSION_KIND = "ols"
DEFAULT_WALD_KIND_NAME = "mean"
DEFAULT_SIMULATION_TARGET = "dgp"
DEFAULT_SIMULATION_OBSERVABLES = True
DEFAULT_SHOCK_SCALE = 1.0
