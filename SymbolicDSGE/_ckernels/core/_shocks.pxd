from libc.stdint cimport int64_t, uint64_t


cdef extern from "shocks.h":
    ctypedef enum native_shock:
        SDSGE_SHOCK_NORMAL = 0
        SDSGE_SHOCK_UNIFORM = 1

    ctypedef struct sdsge_shock_entry:
        native_shock family
        int64_t width
        const int64_t *columns
        const double *factor
        const double *loc
        uint64_t key

    ctypedef struct sdsge_shock_plan:
        const sdsge_shock_entry *entries
        int64_t n_entries
        int64_t T
        int64_t n_exog
        double shock_scale
        int64_t max_width

    int64_t sdsge_shock_scratch_size(
        const sdsge_shock_plan *plan,
    ) noexcept nogil

    void sdsge_shock_draw(
        const sdsge_shock_plan *plan,
        int64_t rep_idx,
        double *scratch,
        double *out,
    ) noexcept nogil


cdef class NativeShockPlan:
    cdef sdsge_shock_plan _plan
    cdef sdsge_shock_entry *_entries
    cdef object _backing

    cdef const sdsge_shock_plan *c_plan(self) noexcept
