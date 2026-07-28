#ifndef SDSGE_MC_CORE_STEPS
#define SDSGE_MC_CORE_STEPS

#include "../_common/sdsge_common.h"
#include "../core/core.h"
#include "../kalman/kalman.h"

/* ``input`` is [A(n,n), B(n,k), x0(n), shock(T,k), params(n_par)].
 * ``simout`` is [states(T,n), observables(T,m)]. */
i64 sdsge_simulate_order1_arena_size(i64 n, i64 k, i64 T, i64 n_par);
void sdsge_simulate_order1_step(f64 *SDSGE_RESTRICT arena,
                                sdsge_measurement_fn measurement, i64 T, i64 n,
                                i64 k, i64 n_par, i64 m,
                                f64 *SDSGE_RESTRICT simout);

/* ``input`` is [hx(nx,nx), gx(ny,nx), bx(nx,n_exog), hxx(nx,nx,nx),
 * gxx(ny,nx,nx), hss(nx), gss(ny), steady_state(nx+ny), x0(nx),
 * shock(T,n_exog), params(n_par), scratch(4*nx + nx*nx)]. ``simout`` is
 * [states(T,nx+ny), observables(T,m)]. */
i64 sdsge_simulate_order2_arena_size(i64 n_state, i64 n_var, i64 n_exog, i64 T,
                                     i64 n_par);
void sdsge_simulate_order2_step(f64 *SDSGE_RESTRICT arena,
                                sdsge_measurement_fn measurement, i64 T, i64 nx,
                                i64 ny, i64 n_exog, i64 n_par, i64 m,
                                f64 *SDSGE_RESTRICT simout);

#endif /* SDSGE_MC_CORE_STEPS */
