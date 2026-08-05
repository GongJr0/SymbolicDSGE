"""Tests for SGU second order assembly.

These cover first order residual consistency, zero tensors for a linear model,
Dynare parity for RBC ``g_xx`` and ``h_xx``, and Dynare parity for the risk
correction terms.

Two Dynare fixtures, one shock each way. rbc_second_order carries the simulation
and IRF goldens; rbc_multishock_second_order carries the three-shock decision
rule, which is the only place the shock cross terms and the correlated risk
correction are pinned.
"""

from __future__ import annotations

import re

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

# ghxu is (n_var, n_state * n_exog) and ghuu (n_var, n_exog^2), both flattened
# the way MATLAB flattens, column major. With one shock that makes ghxu the k
# column [k', z, c] followed by the z column, and ghuu a single [k', z, c].
_DYNARE_GHXU = [
    0.030653832421147185,
    0.0,
    0.0044471781819490769,
    2.382013871953204,
    0.0,
    0.48254743095322666,
]

_DYNARE_GHUU = [2.5073830231086385, 0.0, 0.50794466416128847]

# Row offsets into the DR ordering [k', z, c] shared by all of the above.
_DR_KPRIME, _DR_Z, _DR_C = 0, 1, 2

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


# Dynare's two states are the compiler's lag auxes, so a golden row maps onto our
# tensors with no relabeling. Our third state is the lifted shock, which carries
# what Dynare reports separately as ghxu/ghuu.
_DYNARE_STATES = ("k_lag1", "z_lag1")


def _dynare_pair(dyn_row: list[float]) -> np.ndarray:
    """A Dynare ghxx row [kk, kz, zk, zz] as a 2x2 in state order [k, z]."""
    return np.asarray(dyn_row, dtype=np.float64).reshape(2, 2)


def _our_pair(compiled, tensor_row: np.ndarray) -> np.ndarray:
    """The [k_lag1, z_lag1] sub-block of one row of a state tensor."""
    ix = [compiled.idx[name] for name in _DYNARE_STATES]
    return tensor_row[np.ix_(ix, ix)]


def _our_shock_block(compiled, tensor_row: np.ndarray) -> np.ndarray:
    """The e_st column of one tensor row as Dynare orders it: the two ghxu
    entries [d2/dk de, d2/dz de] followed by ghuu [d2/de2]."""
    e = compiled.idx["e_st"]
    k, z = (compiled.idx[n] for n in _DYNARE_STATES)
    return np.array([tensor_row[k, e], tensor_row[z, e], tensor_row[e, e]])


def _dynare_shock_block(dr_row: int) -> np.ndarray:
    """One decision-rule row's [ghxu_k, ghxu_z, ghuu], to line up with
    ``_our_shock_block``. The two ghxu columns are ``n_var`` apart."""
    n_var = len(_DYNARE_GHUU)
    return np.array(
        [
            _DYNARE_GHXU[dr_row],
            _DYNARE_GHXU[n_var + dr_row],
            _DYNARE_GHUU[dr_row],
        ]
    )


def _levels_steady_state(compiled) -> np.ndarray:
    """The RBC expansion point over the compiled layout, generated vars included."""
    calib = compiled.config.calibration.parameters
    known = {"k": float(calib[sp.Symbol("k_ss")]), "c": float(calib[sp.Symbol("c_ss")])}
    return np.array(
        [known.get(re.sub(r"_lag\d+$", "", n), 0.0) for n in compiled.var_names],
        dtype=np.float64,
    )


def _solved_rbc():
    """The RBC model solved to second order, as (solved, compiled)."""
    model, kalman = ModelParser("tests/fixtures/models/rbc_second_order.yaml").get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()
    return solver.solve(compiled, order=2), compiled


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


def _golden_columns(solved) -> list[int]:
    """Our columns matching the golden's stored [z, k, c].

    The paths were captured against Dynare's own dating, where the reported k is
    the predetermined stock. That is our k_lag1, not our contemporaneous k.
    """
    idx = solved.compiled.idx
    return [idx["z"], idx["k_lag1"], idx["c"]]


def _golden_x0(solved) -> np.ndarray:
    """The Dynare initial condition over our state block [e_st, k_lag1, z_lag1].

    The golden reports z contemporaneously but starts the path one period
    earlier, so the initial lag is one AR step behind the printed value.
    """
    rho = float(solved.compiled.config.calibration.parameters[sp.Symbol("rho")])
    z0, k0 = _DYNARE_SIM_X0
    return np.array([0.0, k0, z0 / rho], dtype=np.float64)


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
    """The golden: our g_xx/h_xx match Dynare's ghxx on the state pair Dynare and
    we share, and the risk correction g_ss/h_ss matches ghs2. The
    independent-solver check on the actual second-order math."""
    model, kalman = ModelParser("tests/fixtures/models/rbc_second_order.yaml").get_all()
    compiled = DSGESolver(model, kalman).compile()
    # States are all compiler-minted: the lifted shock and the two lag auxes.
    assert list(compiled.var_names) == ["e_st", "k_lag1", "z_lag1", "c", "k", "z"]

    layout = ResidualLayout.from_compiled(compiled)
    n_eq, n_state = layout.n_eq, compiled.n_state
    calib = compiled.config.calibration.parameters
    ss = _levels_steady_state(compiled)
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

    # k_lag1(t+1) = k(t), so Dynare's k' row is ours; z_lag1(t+1) = z(t) is linear.
    ctrl = {n: i for i, n in enumerate(compiled.var_names[n_state:])}
    np.testing.assert_allclose(
        _our_pair(compiled, gxx[ctrl["c"]]),
        _dynare_pair(_DYNARE_GHXX_C),
        rtol=5e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        _our_pair(compiled, hxx[compiled.idx["k_lag1"]]),
        _dynare_pair(_DYNARE_GHXX_KPRIME),
        rtol=5e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(hxx[compiled.idx["z_lag1"]], 0.0, atol=1e-12)

    # Risk correction vs ghs2: eta loads the single shock (std sig) on the lifted
    # shock state; x' = h(x) + eta @ eps.
    sig = float(calib[sp.Symbol("sig")])
    eta = np.zeros((n_state, 1), dtype=np.float64)
    eta[compiled.idx["e_st"], 0] = sig
    gss, hss = second_order_risk(a, b, f_xx, gx, gxx, eta, n_state)
    np.testing.assert_allclose(
        [hss[compiled.idx["k_lag1"]], hss[compiled.idx["z_lag1"]], gss[ctrl["c"]]],
        _DYNARE_GHS2,
        rtol=5e-6,
        atol=1e-9,
    )


def test_rbc_shock_state_columns_carry_dynare_ghxu_ghuu():
    """Desugaring lifts the shock into a state, so what Dynare reports as ghxu
    and ghuu lands inside our g_xx/h_xx as the e_st column. That column is a
    third of every tensor row and the ghxx assertions cover none of it.
    """
    solved, compiled = _solved_rbc()
    pol = solved.policy
    ctrl = {n: i for i, n in enumerate(compiled.var_names[compiled.n_state :])}

    np.testing.assert_allclose(
        _our_shock_block(compiled, pol.gxx[ctrl["c"]]),
        _dynare_shock_block(_DR_C),
        rtol=5e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        _our_shock_block(compiled, pol.hxx[compiled.idx["k_lag1"]]),
        _dynare_shock_block(_DR_KPRIME),
        rtol=5e-6,
        atol=2e-7,
    )


def test_rbc_second_order_structural_invariants():
    """What the compiler's own layout forces, independent of any golden.

    The lifted shock is exogenous and i.i.d., so it has no transition row and no
    risk correction. z_lag1(t+1) = z(t) is linear, so its row is flat. And
    k_lag1(t+1) = k(t) makes the k_lag1 state row the same object as the k
    control row, which is the identity that lets a ghxx golden be read off
    either tensor.
    """
    solved, compiled = _solved_rbc()
    pol = solved.policy
    ctrl = {n: i for i, n in enumerate(compiled.var_names[compiled.n_state :])}

    np.testing.assert_allclose(pol.hxx[compiled.idx["e_st"]], 0.0, atol=1e-12)
    np.testing.assert_allclose(pol.hxx[compiled.idx["z_lag1"]], 0.0, atol=1e-12)
    np.testing.assert_allclose(
        pol.hxx[compiled.idx["k_lag1"]], pol.gxx[ctrl["k"]], rtol=1e-12, atol=1e-14
    )
    assert pol.hss[compiled.idx["e_st"]] == 0.0


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

    ss = _levels_steady_state(compiled)
    n_state = compiled.n_state
    ctrl = {n: i for i, n in enumerate(compiled.var_names[n_state:])}
    # steady state was solved/validated to the nonlinear point.
    np.testing.assert_allclose(pol.steady_state, ss, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(
        _our_pair(compiled, pol.gxx[ctrl["c"]]),
        _dynare_pair(_DYNARE_GHXX_C),
        rtol=5e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        _our_pair(compiled, pol.hxx[compiled.idx["k_lag1"]]),
        _dynare_pair(_DYNARE_GHXX_KPRIME),
        rtol=5e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        [
            pol.hss[compiled.idx["k_lag1"]],
            pol.hss[compiled.idx["z_lag1"]],
            pol.gss[ctrl["c"]],
        ],
        _DYNARE_GHS2,
        rtol=5e-6,
        atol=1e-9,
    )
    # First order path is untouched: KleinSolution, no second-order tensors.
    # (levels model -> the expansion point must be supplied; zeros would fail BK.)
    first = solver.solve(compiled, order=1, ss_seed=ss)
    assert first.policy.order == 1
    assert not hasattr(first.policy, "gxx")


def test_rbc_second_order_deterministic_sim_matches_dynare():
    solved = _solve_rbc_second_order()

    out = solved.sim(
        _DYNARE_DETERMINISTIC_SIM.shape[0] - 1,
        x0=_golden_x0(solved),
    )[
        "_X"
    ][:, _golden_columns(solved)]

    np.testing.assert_allclose(
        out,
        _DYNARE_DETERMINISTIC_SIM[1:],
        rtol=2e-6,
        atol=2e-6,
    )


def test_rbc_second_order_stochastic_sim_matches_dynare():
    solved = _solve_rbc_second_order()

    out = solved.sim(
        _DYNARE_STOCHASTIC_SIM.shape[0] - 1,
        x0=_golden_x0(solved),
        shocks={"e": _DYNARE_STOCHASTIC_SHOCKS},
    )["_X"][:, _golden_columns(solved)]

    np.testing.assert_allclose(
        out,
        _DYNARE_STOCHASTIC_SIM[1:],
        rtol=2e-6,
        atol=2e-6,
    )


def test_rbc_second_order_irf_matches_dynare():
    solved = _solve_rbc_second_order()

    out = solved.irf(["e"], T=_DYNARE_IRF.shape[0] - 1)["_X"][
        :, _golden_columns(solved)
    ]

    np.testing.assert_allclose(out, _DYNARE_IRF[1:], rtol=2e-6, atol=2e-6)


# --- three correlated shocks -------------------------------------------------
# The single-shock fixture above cannot reach the shock cross terms, the Cholesky
# branch of the eta builder, or a risk correction against anything but one
# variance. rbc_multishock_second_order.yaml exists for those three.

_MULTISHOCK = "tests/fixtures/models/rbc_multishock_second_order.yaml"
_MS_SHOCK_STATES = ("e_z_st", "e_d_st", "e_g_st")
_MS_STDS = ("sig_z", "sig_d", "sig_g")
_MS_CORRS = {(0, 1): "corr_zd", (0, 2): "corr_zg", (1, 2): "corr_dg"}

# Dynare stoch_simul(order=2) on tests/fixtures/models/rbc_multishock_second_order.mod,
# untouched full precision, as make_rbc_multishock_second_order_goldens.m prints
# it. One list per DR row.
#
# The three vocabularies below are the whole mapping, and are named on our side:
# Dynare's DR rows [k, d, g, z, c], its state columns [k, d, g, z] which are our
# lag auxes, and its exogenous columns [e_z, e_d, e_g] which are our lifted shock
# states. Second-order columns are Kronecker products of those, column (i-1)*q + j
# of kron(A, B) pairing A(i) with B(j): ghxx is state x state, ghxu state x exo,
# ghuu exo x exo.
_MS_DR_ROWS = ("k", "d", "g", "z", "c")
_MS_DR_STATES = ("k_lag1", "d_lag1", "g_lag1", "z_lag1")
_MS_DR_EXO = ("e_z_st", "e_d_st", "e_g_st")

_DYNARE_MS_GHX = [
    [0.97764956105920198, 3.0989174339781824, -0.3378547421051602, 2.0621545726769814],
    [0, 0.80000000000000004, -1.7945246048385654e-18, 0],
    [0, 0, 0.90000000000000013, 0],
    [0, -1.3429187802919278e-16, -4.5749892275358046e-18, 0.94999999999999973],
    [
        0.032451449658549333,
        -3.0989174339781824,
        -0.11214525789483983,
        0.80240672562080839,
    ],
]

_DYNARE_MS_GHU = [
    [2.1706890238705081, 3.8736467924727291, -0.37539415789462244],
    [0, 0.99999999999999989, 0],
    [-0, -0, 1],
    [1, -0, -0],
    [0.8446386585482194, -3.8736467924727291, -0.12460584210537756],
]

_DYNARE_MS_GHXX = [
    [
        -0.00027449463044743234,
        0.062557461268933023,
        -0.00096642240723384819,
        0.029666628209446988,
        0.062557461268933023,
        -4.7511056132642189,
        -0.20841866901845763,
        1.3867403480517568,
        -0.00096642240723384841,
        -0.2084186690184576,
        -0.34196846880008797,
        -0.013242044372532652,
        0.029666628209446988,
        1.3867403480517571,
        -0.013242044372532638,
        2.2929612765158933,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        1.2779006356522617e-18,
        1.0223205085218094e-17,
        0,
        1.2779006356522617e-18,
        -2.6727647100921956e-51,
        0,
        0,
        1.0223205085218094e-17,
        0,
        0,
    ],
    [0] * 16,
    [0] * 16,
    [
        -0.0005550993496256703,
        -0.062557461268933023,
        0.00096642240723384841,
        0.0036793319724172271,
        -0.062557461268933037,
        4.7511056132642171,
        0.20841866901845768,
        -1.3867403480517575,
        0.00096642240723384862,
        0.20841866901845757,
        -0.063031531199912214,
        0.013242044372532638,
        0.0036793319724172276,
        -1.3867403480517575,
        0.013242044372532628,
        0.42837195686700674,
    ],
]

_DYNARE_MS_GHXU = [
    [
        0.031228029694154731,
        0.078196826586166349,
        -0.0010738026747042775,
        1.459726682159743,
        -5.9388820165802807,
        -0.23157629890939729,
        -0.013938994076350221,
        -0.26052333627307239,
        -0.37996496533343094,
        2.4136434489640988,
        1.7334254350646985,
        -0.014713382636147422,
    ],
    [
        1.5973757945653272e-19,
        -0,
        -3.9934394864133179e-20,
        -0,
        -0,
        -0,
        6.3895031782613087e-19,
        -0,
        -2.5558012713045235e-18,
        -0,
        -8.1785640681744752e-17,
        -0,
    ],
    [-0, 0, 0, 0, 0, 0, 0, 0, 0, -0, 0, 0],
    [0] * 12,
    [
        0.0038729810235970858,
        -0.078196826586166349,
        0.0010738026747042775,
        -1.4597266821597434,
        5.9388820165802807,
        0.23157629890939729,
        0.013938994076350207,
        0.26052333627307239,
        -0.070035034666569115,
        0.45091784933369128,
        -1.7334254350646985,
        0.014713382636147422,
    ],
]

_DYNARE_MS_GHUU = [
    [
        2.5406773146990527,
        1.8246583526996811,
        -0.015487771195944695,
        1.8246583526996811,
        -7.423602520725356,
        -0.28947037363674705,
        -0.015487771195944709,
        -0.28947037363674688,
        -0.42218329481492323,
    ],
    [
        -0,
        -0,
        6.3895031782613087e-19,
        -0,
        3.2714256272697901e-16,
        -1.0223205085218094e-17,
        -0,
        1.0223205085218094e-17,
        -0,
    ],
    [0] * 9,
    [0] * 9,
    [
        0.47465036771967467,
        -1.8246583526996811,
        0.015487771195944695,
        -1.8246583526996811,
        7.423602520725356,
        0.28947037363674705,
        0.015487771195944709,
        0.28947037363674688,
        -0.077816705185076787,
    ],
]

_DYNARE_MS_GHS2 = [
    0.031518725236580371,
    -3.9934394864133179e-20,
    0.0,
    0.0,
    -0.031518725236580371,
]


def _solved_multishock():
    """The three-shock RBC solved to second order, as (solved, compiled)."""
    model, kalman = ModelParser(_MULTISHOCK).get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()
    return solver.solve(compiled, order=2), compiled


def _ms_preproc(solved, compiled):
    """(a, b, f_xx) at the solved steady state, enough to re-run the risk
    correction with an eta of our choosing."""
    par = np.array(
        [
            float(compiled.config.calibration.parameters[p])
            for p in compiled.calib_params
        ],
        dtype=np.float64,
    )
    n_eq = len(compiled.var_names)
    ss = solved.policy.steady_state
    cf = compiled.construct_objective_cfunc()
    cf_bc = compiled.construct_objective_cfunc_bicomplex()
    a, b = klein_preprocess(cf.address, ss, par, n_eq, False)
    return a, b, bicomplex_hessian(cf_bc.address, ss, par, n_eq)


def _ms_covariance(compiled) -> np.ndarray:
    """The 3x3 innovation covariance the yaml calibrates, read straight off the
    parameters rather than off eta."""
    calib = compiled.config.calibration.parameters
    stds = np.array([float(calib[sp.Symbol(s)]) for s in _MS_STDS])
    corr = np.eye(3)
    for (i, j), name in _MS_CORRS.items():
        corr[i, j] = corr[j, i] = float(calib[sp.Symbol(name)])
    return corr * np.outer(stds, stds)


def _ms_dr_row(solved, compiled, dr_row: str):
    """(first order, second order, risk) for one Dynare DR row.

    Dynare's k row is next period's capital, which on our side is the k_lag1
    transition row; every other DR row is one of our controls."""
    pol = solved.policy
    if dr_row == "k":
        i = compiled.idx["k_lag1"]
        return np.real(pol.p)[i], pol.hxx[i], pol.hss[i]
    ctrl = {n: i for i, n in enumerate(compiled.var_names[compiled.n_state :])}
    i = ctrl[dr_row]
    return np.real(pol.f)[i], pol.gxx[i], pol.gss[i]


def test_multishock_first_order_matches_dynare():
    """ghx and ghu with three shocks. The lifted shock states hold ghu, so the
    two Dynare arrays are one slice of our first order rule each."""
    solved, compiled = _solved_multishock()
    si = [compiled.idx[n] for n in _MS_DR_STATES]
    ei = [compiled.idx[n] for n in _MS_DR_EXO]

    for r, name in enumerate(_MS_DR_ROWS):
        first, _, _ = _ms_dr_row(solved, compiled, name)
        np.testing.assert_allclose(
            first[si], _DYNARE_MS_GHX[r], rtol=5e-6, atol=2e-7, err_msg=f"ghx {name}"
        )
        np.testing.assert_allclose(
            first[ei], _DYNARE_MS_GHU[r], rtol=5e-6, atol=2e-7, err_msg=f"ghu {name}"
        )


def test_multishock_second_order_matches_dynare():
    """ghxx, ghxu and ghuu with three shocks, every DR row.

    ghxu and ghuu are the payload: with one shock they are a rescaling of ghxx
    and constrain nothing new, but here they carry the cross terms between
    distinct innovations, which no single-shock model has.
    """
    solved, compiled = _solved_multishock()
    si = [compiled.idx[n] for n in _MS_DR_STATES]
    ei = [compiled.idx[n] for n in _MS_DR_EXO]
    ns, ne = len(si), len(ei)

    for r, name in enumerate(_MS_DR_ROWS):
        _, tensor, _ = _ms_dr_row(solved, compiled, name)
        for label, ix, golden, shape in (
            ("ghxx", (si, si), _DYNARE_MS_GHXX[r], (ns, ns)),
            ("ghxu", (si, ei), _DYNARE_MS_GHXU[r], (ns, ne)),
            ("ghuu", (ei, ei), _DYNARE_MS_GHUU[r], (ne, ne)),
        ):
            np.testing.assert_allclose(
                tensor[np.ix_(*ix)],
                np.asarray(golden, dtype=np.float64).reshape(shape),
                rtol=5e-6,
                atol=2e-7,
                err_msg=f"{label} {name}",
            )


def test_multishock_risk_correction_matches_dynare():
    """ghs2 against a full covariance rather than one variance. This is the only
    assertion in the file that the shock correlations reach the solution at all,
    since eta enters nothing else."""
    solved, compiled = _solved_multishock()
    ours = [_ms_dr_row(solved, compiled, name)[2] for name in _MS_DR_ROWS]

    np.testing.assert_allclose(ours, _DYNARE_MS_GHS2, rtol=5e-6, atol=1e-8)


def test_multishock_eta_reproduces_the_calibrated_covariance():
    """eta is a Cholesky factor, so only ``eta @ eta.T`` is meaningful, and that
    has to be the calibrated covariance on the exog-state rows and zero below."""
    _, compiled = _solved_multishock()
    eta = DSGESolver._build_eta(compiled)
    n_exog = compiled.n_exog

    assert eta.shape == (compiled.n_state, n_exog)
    np.testing.assert_allclose(
        eta[:n_exog] @ eta[:n_exog].T, _ms_covariance(compiled), rtol=1e-13, atol=0.0
    )
    np.testing.assert_array_equal(eta[n_exog:], 0.0)


def test_multishock_risk_correction_reads_the_off_diagonals():
    """The correlations have to reach g_ss, and only through ``eta @ eta.T``.

    Zeroing the off-diagonals moves g_ss, so the covariance is not being read as
    a diagonal. Refactoring the same covariance through its symmetric square root
    instead of its Cholesky leaves g_ss alone, so nothing is reading the factor.
    """
    solved, compiled = _solved_multishock()
    a, b, f_xx = _ms_preproc(solved, compiled)
    gxx = solved.policy.gxx
    n_state, n_exog = compiled.n_state, compiled.n_exog

    eta = DSGESolver._build_eta(compiled)
    cov = _ms_covariance(compiled)
    gx = np.real(solved.policy.f)

    diag = np.zeros_like(eta)
    diag[:n_exog, :] = np.diag(np.sqrt(np.diag(cov)))
    w, v = np.linalg.eigh(cov)
    root = np.zeros_like(eta)
    root[:n_exog, :] = v @ np.diag(np.sqrt(w)) @ v.T

    gss = second_order_risk(a, b, f_xx, gx, gxx, eta, n_state)[0]
    gss_diag = second_order_risk(a, b, f_xx, gx, gxx, diag, n_state)[0]
    gss_root = second_order_risk(a, b, f_xx, gx, gxx, root, n_state)[0]

    assert not np.allclose(gss, gss_diag, rtol=1e-3, atol=1e-12)
    np.testing.assert_allclose(gss, gss_root, rtol=1e-10, atol=1e-15)


def test_multishock_second_order_structural_invariants():
    """The same layout facts the single-shock fixture pins, once per shock.

    Every lifted shock is exogenous and i.i.d., so it has no transition row and
    no risk correction; every AR(1) lag row is linear, so it is flat; and
    k_lag1(t+1) = k(t) keeps the k_lag1 state row equal to the k control row.
    """
    solved, compiled = _solved_multishock()
    pol = solved.policy
    ctrl = {n: i for i, n in enumerate(compiled.var_names[compiled.n_state :])}

    np.testing.assert_allclose(pol.gxx, pol.gxx.transpose(0, 2, 1), rtol=0, atol=0)
    np.testing.assert_allclose(pol.hxx, pol.hxx.transpose(0, 2, 1), rtol=0, atol=0)

    linear = _MS_SHOCK_STATES + ("z_lag1", "d_lag1", "g_lag1")
    for name in linear:
        np.testing.assert_allclose(pol.hxx[compiled.idx[name]], 0.0, atol=1e-12)
    for name in _MS_SHOCK_STATES:
        assert pol.hss[compiled.idx[name]] == 0.0

    np.testing.assert_allclose(
        pol.hxx[compiled.idx["k_lag1"]], pol.gxx[ctrl["k"]], rtol=1e-12, atol=1e-14
    )
