"""Monte Carlo per-replication data sources and the reference Kalman filter.

Shared contract for the factories in this group:

- These seed a pipeline: a DATAGEN step (``simulation`` or ``raw_data``) must
  run first to populate ``context.data`` for that replication.
- Output location: datagen fills ``context.data`` (states / observables / raw
  series); the ``filter`` step stores a ``FilterResult`` that downstream steps
  read via ``source="filter"``.
"""

from .builtins import (
    raw_data_step,
    reference_filter_step,
    simulation_step,
)

__all__ = [
    "simulation_step",
    "raw_data_step",
    "reference_filter_step",
]
