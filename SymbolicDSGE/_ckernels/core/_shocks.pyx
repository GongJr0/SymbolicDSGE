# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
from enum import IntEnum, unique

from libc.stdint cimport int64_t, uint64_t
from cpython.mem cimport PyMem_Malloc, PyMem_Free


@unique
class ShockCode(IntEnum):
    """Integer codes the native draw dispatches an entry on.

    The ``SDSGE_SHOCK_*`` values from ``shocks.h``, exported
    and enum-ified.
    """

    PATH = native_shock.SDSGE_SHOCK_PATH
    NORMAL = native_shock.SDSGE_SHOCK_NORMAL
    UNIFORM = native_shock.SDSGE_SHOCK_UNIFORM
    STUDENT_T = native_shock.SDSGE_SHOCK_STUDENT_T

    @classmethod
    def for_dist(cls, dist):
        """Return a code when the native kernel supports the family."""
        if not isinstance(dist, str):
            return None
        if dist == "norm":
            return cls.NORMAL
        if dist == "uni":
            return cls.UNIFORM
        if dist == "t":
            return cls.STUDENT_T
        return None


cdef class NativeShockPlan:
    """A shock spec resolved into the layout the native draw reads.

    Owns the C entry array and holds a reference to every NumPy buffer its
    entries point at, so the plan is the single lifetime anchor: a simulation
    step keeps one of these alive for as long as its context references it.

    The plan is immutable and shared read-only across workers. Nothing here is
    touched during the run, which is what lets the draw be reentrant.
    """

    def __cinit__(self):
        self._entries = NULL
        self._backing = []

    def __dealloc__(self):
        if self._entries != NULL:
            PyMem_Free(self._entries)
            self._entries = NULL

    cdef const sdsge_shock_plan *c_plan(self) noexcept:
        return &self._plan

    @property
    def scratch_size(self):
        """Extra float arena elements the draw needs, for step sizing."""
        return sdsge_shock_plan_arena_size(&self._plan).n_float

    @property
    def n_entries(self):
        return self._plan.n_entries

    def fill(self, double[:, ::1] out, int64_t rep_idx):
        """Draw one replication into a caller-owned ``(T, n_exog)`` block."""
        if rep_idx < 0:
            raise ValueError("rep_idx must be non-negative.")
        if out.shape[0] != self._plan.T or out.shape[1] != self._plan.n_exog:
            raise ValueError(
                f"Shock block must have shape ({self._plan.T}, "
                f"{self._plan.n_exog}); got ({out.shape[0]}, {out.shape[1]})."
            )
        cdef double[::1] scratch = np.empty(
            max(sdsge_shock_plan_arena_size(&self._plan).n_float, 1), dtype=np.float64
        )
        with nogil:
            sdsge_shock_draw(&self._plan, rep_idx, &scratch[0], &out[0, 0])

    def draw(self, int64_t rep_idx):
        """Materialize one replication's ``(T, n_exog)`` shock block.

        This is the route makeing a single replication reproducible outside an
        MC pipeline and lets sim use a native plan to draw.
        """
        cdef double[:, ::1] out = np.zeros(
            (self._plan.T, self._plan.n_exog), dtype=np.float64
        )
        self.fill(out, rep_idx)
        return np.asarray(out)


def native_shock_plan(
    object entries,
    int64_t T,
    int64_t n_exog,
    double shock_scale,
):
    """Build a native shock plan from resolved entries.

    Each entry is ``(family, columns, factor, loc, params, key)``. ``family`` is
    one of the native shock family codes and selects both the standardized
    variate the draw fills and the ``params`` member it reads; every other field
    is read by every family. ``columns`` is the
    int64 array of exogenous column indices the entry drives, in the order its
    ``factor`` was built in. ``factor`` is the row-major ``(width, width)``
    matrix with ``factor @ factor.T`` equal to the covariance, a 1x1 holding the
    standard deviation at width 1. ``loc`` is the width-long location.

    The resolver settles the shapes. Buffers are converted to contiguous arrays
    of the required dtype here; the memoryviews enforce their ranks. A shape
    mismatch is a lowering bug rather than user input.

    The first canonical column index separates entries sharing a seed;
    the replication index selects each entry's replication stream.
    """
    cdef NativeShockPlan plan = NativeShockPlan()
    cdef int64_t n = len(entries)
    cdef int64_t i
    cdef int64_t[::1] columns_mv
    cdef double[:, ::1] factor_mv
    cdef double[::1] loc_mv
    cdef double[:, ::1] path_mv
    plan._entries = <sdsge_shock_entry *>PyMem_Malloc(
        <size_t>n * sizeof(sdsge_shock_entry)
    )
    if plan._entries == NULL:
        raise MemoryError("Could not allocate native shock entries.")

    for i, e in enumerate(entries):
        columns_mv = np.ascontiguousarray(e.indices, dtype=np.int64)
        plan._backing.append(columns_mv)

        plan._entries[i].family = <native_shock>e.family
        plan._entries[i].width = e.width
        plan._entries[i].columns = &columns_mv[0]

        if e.family == native_shock.SDSGE_SHOCK_PATH:
            if e.value.shape[0] != T:
                raise ValueError(
                    f"Path period length {e.value.shape[0]} does not match T={T}."
                )
            path = np.ascontiguousarray(e.value, dtype=np.float64)
            plan._backing.append(path)
            path_mv = path

            plan._entries[i].path = &path_mv[0, 0]
            plan._entries[i].factor = NULL
            plan._entries[i].loc = NULL
            plan._entries[i].key = 0
            continue

        factor_mv = np.ascontiguousarray(e.factor, dtype=np.float64)
        plan._backing.append(factor_mv)
        plan._entries[i].factor = &factor_mv[0, 0]

        loc_mv = np.ascontiguousarray(e.loc, dtype=np.float64)
        plan._backing.append(loc_mv)
        plan._entries[i].loc = &loc_mv[0]
        plan._entries[i].path = NULL

        # `family` selects the union member the kernel reads. A spec's kwargs
        # are checked against its family when it binds, so the key is here.
        if e.family == native_shock.SDSGE_SHOCK_STUDENT_T:
            plan._entries[i].params.df = <double>e.kwargs["df"]

        plan._entries[i].key = <uint64_t>e._native_seed_key

    plan._plan.entries = plan._entries
    plan._plan.n_entries = n
    plan._plan.T = T
    plan._plan.n_exog = n_exog
    plan._plan.shock_scale = shock_scale
    return plan
