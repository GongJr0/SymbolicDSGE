from libc.stdint cimport int64_t, uint64_t

cdef extern from "../_common/sdsge_common.h":
    ctypedef struct arena_size:
        int64_t n_float
        int64_t n_int

cdef extern from "../rng/philox.h":
    ctypedef struct sdsge_beta_params:
        double a
        double b

    ctypedef union sdsge_sampler_params:
        double df
        double a
        sdsge_beta_params beta

cdef extern from "shocks.h":
    ctypedef enum native_shock:
        SDSGE_SHOCK_PATH = -1
        SDSGE_SHOCK_NORMAL = 0
        SDSGE_SHOCK_UNIFORM = 1
        SDSGE_SHOCK_STUDENT_T = 2
        SDSGE_SHOCK_EXPONENTIAL = 3
        SDSGE_SHOCK_GAMMA = 4
        SDSGE_SHOCK_BETA = 5

    ctypedef struct sdsge_shock_entry:
        native_shock family
        const double *path
        int64_t width
        const int64_t *columns
        const double *factor
        const double *loc
        sdsge_sampler_params params
        uint64_t key

    ctypedef struct sdsge_shock_plan:
        const sdsge_shock_entry *entries
        int64_t n_entries
        int64_t T
        int64_t n_exog
        double shock_scale

    arena_size sdsge_shock_plan_arena_size(
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
