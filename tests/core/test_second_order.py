"""Tests for SGU second order assembly.

These cover first order residual consistency, zero tensors for a linear model,
Dynare parity for RBC ``g_xx`` and ``h_xx``, and Dynare parity for the risk
correction terms.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from SymbolicDSGE._ckernels.core._core import bicomplex_hessian, klein_preprocess
from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.solver_backend import klein_solve
from SymbolicDSGE._symbolic_printers import ResidualLayout
from SymbolicDSGE._ckernels.core import (
    second_order,
    second_order_risk,
)
from _oracles.core import first_order_residual

# Dynare stoch_simul(order=2) on tests/fixtures/models/rbc_second_order.mod,
# untouched full precision. Rows are DR order [k', z, c]; the four
# columns are the state-pair second derivatives in Dynare's state order [k, z],
# i.e. [kk, kz, zk, zz].
_DYNARE_GHXX_KPRIME = [
    -0.00020831558951371761,
    0.029121140800089713,
    0.029121140800089713,
    2.2629131783555407,
]
_DYNARE_GHXX_C = [
    -0.0006212783838050477,
    0.004224819272851745,
    0.004224819272851745,
    0.45842005940556829,
]
# ghs2 is the sigma squared risk correction in DR order [k', z, c].
_DYNARE_GHS2 = [0.0010614857740643515, 0.0, -0.0010614857740643515]
_DYNARE_SIM_X0 = np.array(
    [0.020000000000000004, 28.631902640387651],
    dtype=np.float64,
)
_DYNARE_STOCHASTIC_SHOCKS = np.array(
    [
        0.0025,
        -0.006999999999999999,
        0.001,
        0.006,
        -0.002,
        0.0,
        0.0035,
        -0.0045000000000000005,
        0.0015,
        -0.001,
        0.0005,
        0.002,
    ],
    dtype=np.float64,
)
_DYNARE_DETERMINISTIC_SIM = np.array(
    [
        [0.020000000000000004, 28.631902640387651, 2.3331413462184378],
        [0.019000000000000006, 28.66931729602415, 2.3336280327710504],
        [0.018050000000000007, 28.703554206124828, 2.3340431325038771],
        [0.017147500000000006, 28.734805296825567, 2.3343915505341153],
        [0.016290125000000006, 28.763252191959324, 2.3346778999434679],
        [0.015475618750000007, 28.789066725890553, 2.3349065189666307],
        [0.014701837812500007, 28.812411433235763, 2.3350814871078369],
        [0.013966745921875008, 28.833440016284651, 2.3352066402595728],
        [0.013268408625781259, 28.852297790933964, 2.3352855848918375],
        [0.012604988194492198, 28.869122111939792, 2.3353217113750402],
        [0.011974738784767588, 28.884042778284527, 2.3353182064947986],
        [0.011376001845529209, 28.897182419442043, 2.3352780652125178],
        [0.010807201753252749, 28.908656863310128, 2.335204101721549],
        [0.010266841665590113, 28.918575486562403, 2.335098959845062],
        [0.0097534995823106083, 28.927041548153856, 2.3349651228183435],
        [0.0092658246031950778, 28.93415250669478, 2.3348049224951315],
        [0.0088025333730353244, 28.940000322387565, 2.3346205480147262],
        [0.0083624067043835595, 28.944671744199891, 2.3344140539639975],
        [0.0079442863691643814, 28.948248582926407, 2.3341873680659777],
        [0.0075470720507061625, 28.950807970769276, 2.3339422984245233],
        [0.0071697184481708549, 28.95242260804617, 2.3336805403524536],
        [0.0068112325257623126, 28.953160997612436, 2.3334036828087119],
        [0.0064706708994741978, 28.953087667562468, 2.3331132144683409],
        [0.0061471373545004884, 28.952263382753838, 2.3328105294474453],
        [0.0058397804867754647, 28.95074534567657, 2.3324969327038585],
    ],
    dtype=np.float64,
)
_DYNARE_STOCHASTIC_SIM = np.array(
    [
        [0.020000000000000004, 28.631902640387651, 2.3331413462184378],
        [0.021500000000000005, 28.66931729602415, 2.3357562251978305],
        [0.013425000000000006, 28.709145038467859, 2.3303154260015293],
        [0.013753750000000006, 28.729958627657929, 2.3313369510589834],
        [0.019066062500000008, 28.7509733672509, 2.3366003465834702],
        [0.016112759375000008, 28.78329573963504, 2.3352420420367257],
        [0.015307121406250008, 28.808206724598033, 2.3354456811353068],
        [0.018041765335937508, 28.830690221770812, 2.3385717101455277],
        [0.012639677069140633, 28.858707045916997, 2.3349803615426126],
        [0.013507693215683601, 28.873970175652339, 2.3362602017573062],
        [0.011832308554899421, 28.890775262464391, 2.3354369553141279],
        [0.011740693127154452, 28.903426784502006, 2.33580937414292],
        [0.01315365847079673, 28.915552039589862, 2.3374393839900445],
    ],
    dtype=np.float64,
)
_DYNARE_IRF = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.01, 0.0, 0.0084210904641053297],
        [0.0095000000000000015, 0.021882952792957155, 0.0087840073237601679],
        [0.0090250000000000018, 0.042108724011733045, 0.0091083438671692996],
        [0.0085737500000000015, 0.060774389224906855, 0.0093965854610504884],
        [0.0081450625000000013, 0.077971849790870351, 0.0096510754976724122],
        [0.0077378093750000015, 0.093788094000689881, 0.0098740233115290899],
        [0.0073509189062500018, 0.10830544577091672, 0.010067511641409066],
        [0.0069833729609375022, 0.12160180142023336, 0.010233503665953325],
        [0.0066342043128906278, 0.13375085504719664, 0.01037384963887833],
        [0.0063024940972460971, 0.14482231300926429, 0.010490293148242458],
        [0.0059873693923837923, 0.15488209798637342, 0.010584477022491878],
        [0.0056880009227646029, 0.16399254309524025, 0.010657948904508441],
        [0.0054036008766263728, 0.17221257650358979, 0.01071216651345841],
        [0.0051334208327950548, 0.17959789697684414, 0.010748502612957456],
        [0.0048767497911553024, 0.18620114077317496, 0.010768249702833188],
        [0.0046329123015975372, 0.19207204028658609, 0.010772624450687829],
        [0.0044012666865176605, 0.19725757482179773, 0.010762771878388477],
        [0.004181203352191778, 0.20180211386903935, 0.010739769317669179],
        [0.003972143184582189, 0.20574755323176319, 0.010704630148135497],
        [0.00377353602535308, 0.20913344434544712, 0.010658307330125627],
        [0.0035848592240854261, 0.21199711711127378, 0.010601696744113376],
        [0.003405616262881155, 0.21437379655460376, 0.01053564034761667],
        [0.0032353354497370976, 0.21629671360465608, 0.010460929159905152],
        [0.0030735686772502429, 0.21779721027871446, 0.010378306084176891],
    ],
    dtype=np.float64,
)


def _dynare_to_our_convention(dyn_row: list[float], rho: float) -> np.ndarray:
    """Map a Dynare ghxx row ([kk, kz, zk, zz] in state order [k, z]) to our 2x2
    Hessian in state order [z, k]. Our offset-0/+1 timing puts the innovation on
    z(t+1) vs Dynare's z(t), so our z-state = rho * Dynare's -> every z-derivative
    carries 1/rho (k is unaffected: same predetermined stock in both datings)."""
    kk, kz, _zk, zz = dyn_row
    return np.array([[zz / rho**2, kz / rho], [kz / rho, kk]])


def _drive(path):
    """Compile a model and run the full second-order preproc chain at ss = 0."""
    model, kalman = ModelParser(path).get_all()
    compiled = DSGESolver(model, kalman).compile()
    layout = ResidualLayout.from_compiled(compiled)
    n_eq, n_state = layout.n_eq, compiled.n_state

    ss = np.zeros(layout.n_var, dtype=np.float64)
    par = np.array(
        [
            float(compiled.config.calibration.parameters[p])
            for p in compiled.calib_params
        ],
        dtype=np.float64,
    )
    cf = compiled.construct_objective_cfunc()
    cf_bc = compiled.construct_objective_cfunc_bicomplex()

    a, b = klein_preprocess(cf.address, ss, par, n_eq, False)
    sol = klein_solve(cf, par, ss, n_state)
    gx, hx = np.real(sol.f), np.real(sol.p)
    f_xx = bicomplex_hessian(cf_bc.address, ss, par, n_eq)
    return a, b, f_xx, gx, hx, n_state


def _solve_rbc_second_order():
    model, kalman = ModelParser("tests/fixtures/models/rbc_second_order.yaml").get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()
    return solver.solve(compiled, order=2)


@pytest.mark.parametrize("path", ["MODELS/test.yaml", "MODELS/POST82.yaml"])
def test_first_order_foc_holds(path):
    a, b, _f_xx, gx, hx, n_state = _drive(path)
    foc = first_order_residual(a, b, gx, hx, n_state)
    np.testing.assert_allclose(foc, 0.0, atol=1e-8)


@pytest.mark.parametrize("path", ["MODELS/test.yaml", "MODELS/POST82.yaml"])
def test_linear_model_has_zero_second_order(path):
    a, b, f_xx, gx, hx, n_state = _drive(path)
    gxx, hxx = second_order(a, b, f_xx, gx, hx, n_state)

    nx = n_state
    ny = gx.shape[0]
    assert gxx.shape == (ny, nx, nx)
    assert hxx.shape == (nx, nx, nx)
    np.testing.assert_allclose(gxx, 0.0, atol=1e-6)
    np.testing.assert_allclose(hxx, 0.0, atol=1e-6)


def test_rbc_second_order_matches_dynare():
    """The golden: our g_xx/h_xx match Dynare's ghxx (after the [k,z]->[z,k]
    reorder + the (1/rho)^m timing map), and the risk correction g_ss/h_ss matches
    ghs2 directly (a constant, so no timing factor). The independent-solver check
    on the actual second-order math."""
    model, kalman = ModelParser("tests/fixtures/models/rbc_second_order.yaml").get_all()
    compiled = DSGESolver(model, kalman).compile()
    assert list(compiled.var_names) == ["z", "k", "c"]  # states [z, k], control c

    layout = ResidualLayout.from_compiled(compiled)
    n_eq, n_state = layout.n_eq, compiled.n_state
    calib = compiled.config.calibration.parameters
    rho = float(calib[sp.Symbol("rho")])
    ss_map = {
        "z": 0.0,
        "k": float(calib[sp.Symbol("k_ss")]),
        "c": float(calib[sp.Symbol("c_ss")]),
    }
    ss = np.array([ss_map[nm] for nm in compiled.var_names], dtype=np.float64)
    par = np.array([float(calib[p]) for p in compiled.calib_params], dtype=np.float64)

    cf = compiled.construct_objective_cfunc()
    cf_bc = compiled.construct_objective_cfunc_bicomplex()
    eq = compiled.equations

    # Steady state actually clears the residual.
    resid = eq(
        ss.astype(np.complex128), ss.astype(np.complex128), par.astype(np.complex128)
    )
    np.testing.assert_allclose(np.real(resid), 0.0, atol=1e-7)

    a, b = klein_preprocess(cf.address, ss, par, n_eq, False)
    sol = klein_solve(cf, par, ss, n_state)
    assert sol.stab == 0
    gx, hx = np.real(sol.f), np.real(sol.p)
    f_xx = bicomplex_hessian(cf_bc.address, ss, par, n_eq)
    gxx, hxx = second_order(a, b, f_xx, gx, hx, n_state)

    # gxx[0] = c (the single control); hxx[1] = k' (state index 1); hxx[0] = z'.
    np.testing.assert_allclose(
        gxx[0], _dynare_to_our_convention(_DYNARE_GHXX_C, rho), rtol=1e-4, atol=1e-6
    )
    np.testing.assert_allclose(
        hxx[1],
        _dynare_to_our_convention(_DYNARE_GHXX_KPRIME, rho),
        rtol=1e-4,
        atol=1e-6,
    )
    np.testing.assert_allclose(hxx[0], 0.0, atol=1e-6)  # z' is linear

    # Risk correction vs ghs2: eta loads the single shock (std sig) on z (state 0);
    # x' = h(x) + eta @ eps. There is no timing factor because g_ss and h_ss
    # are constants.
    sig = float(calib[sp.Symbol("sig")])
    eta = np.zeros((n_state, 1), dtype=np.float64)
    eta[0, 0] = sig
    gss, hss = second_order_risk(a, b, f_xx, gx, gxx, eta, n_state)
    # ours [g_ss(c); h_ss(z', k')] -> Dynare DR order [k', z, c]
    np.testing.assert_allclose(
        [hss[1], hss[0], gss[0]], _DYNARE_GHS2, rtol=1e-4, atol=1e-8
    )


def test_solve_order2_wiring():
    """The .solve(order=2) public path end to end: it resolves + cross-checks the
    nonlinear steady state, builds eta from the shock calibration, and returns a
    PerturbationSolution whose tensors match the Dynare goldens. order=1 is
    unchanged (a plain KleinSolution with no second-order fields)."""
    model, kalman = ModelParser("tests/fixtures/models/rbc_second_order.yaml").get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()

    solved = solver.solve(compiled, order=2)
    pol = solved.policy
    assert pol.order == 2

    calib = compiled.config.calibration.parameters
    rho = float(calib[sp.Symbol("rho")])
    k_ss, c_ss = float(calib[sp.Symbol("k_ss")]), float(calib[sp.Symbol("c_ss")])
    # steady state was solved/validated to the nonlinear point (order [z, k, c]).
    np.testing.assert_allclose(
        pol.steady_state, [0.0, k_ss, c_ss], rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(
        pol.gxx[0], _dynare_to_our_convention(_DYNARE_GHXX_C, rho), rtol=1e-4, atol=1e-6
    )
    np.testing.assert_allclose(
        pol.hxx[1],
        _dynare_to_our_convention(_DYNARE_GHXX_KPRIME, rho),
        rtol=1e-4,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        [pol.hss[1], pol.hss[0], pol.gss[0]], _DYNARE_GHS2, rtol=1e-4, atol=1e-8
    )
    # First order path is untouched: KleinSolution, no second-order tensors.
    # (levels model -> the expansion point must be supplied; zeros would fail BK.)
    first = solver.solve(compiled, order=1, steady_state=[0.0, k_ss, c_ss])
    assert first.policy.order == 1
    assert not hasattr(first.policy, "gxx")


def test_rbc_second_order_deterministic_sim_matches_dynare():
    solved = _solve_rbc_second_order()

    out = solved.sim(
        _DYNARE_DETERMINISTIC_SIM.shape[0] - 1,
        x0=_DYNARE_SIM_X0,
    )["_X"]

    np.testing.assert_allclose(
        out,
        _DYNARE_DETERMINISTIC_SIM,
        rtol=2e-6,
        atol=2e-6,
    )


def test_rbc_second_order_stochastic_sim_matches_dynare():
    solved = _solve_rbc_second_order()

    out = solved.sim(
        _DYNARE_STOCHASTIC_SIM.shape[0] - 1,
        x0=_DYNARE_SIM_X0,
        shocks={"z": _DYNARE_STOCHASTIC_SHOCKS},
    )["_X"]

    np.testing.assert_allclose(
        out,
        _DYNARE_STOCHASTIC_SIM,
        rtol=2e-6,
        atol=2e-6,
    )


def test_rbc_second_order_irf_matches_dynare():
    solved = _solve_rbc_second_order()

    out = solved.irf(["z"], T=_DYNARE_IRF.shape[0] - 1)["_X"]

    np.testing.assert_allclose(out, _DYNARE_IRF, rtol=2e-6, atol=2e-6)


def test_solve_order2_rejects_inconsistent_steady_state():
    """A steady_state= that does not clear F(ss, ss) is rejected rather than used
    as a bad expansion point."""
    model, kalman = ModelParser("tests/fixtures/models/rbc_second_order.yaml").get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()
    with pytest.raises(ValueError, match="disagrees with the numerically solved"):
        solver.solve(compiled, order=2, steady_state=[0.0, 1.0, 1.0])
