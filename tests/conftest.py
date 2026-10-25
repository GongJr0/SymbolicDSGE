from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

TESTS_DIR = Path(__file__).resolve().parent
MODELS_DIR = TESTS_DIR / "fixtures" / "models"

TEST_MODEL_PATH = MODELS_DIR / "test.yaml"
POST82_TEST_MODEL_PATH = MODELS_DIR / "POST82.yaml"
DENSE_LKJ_TEST_MODEL_PATH = MODELS_DIR / "LKJ_DENSE.yaml"
RBC_SECOND_ORDER_TEST_MODEL_PATH = MODELS_DIR / "rbc_second_order.yaml"
RBC_MULTISHOCK_TEST_MODEL_PATH = MODELS_DIR / "rbc_multishock_second_order.yaml"
RBC_OCCBIN_TEST_MODEL_PATH = MODELS_DIR / "rbc_occbin.yaml"
NK_ZLB_TEST_MODEL_PATH = MODELS_DIR / "nk_zlb_1_constraint.yaml"
#: Three shocks with no named correlation, the unnamed-pair case estimation rejects.
GALI_2015_TEST_MODEL_PATH = MODELS_DIR / "gali_2015.yaml"

#: Names an indirect ``parametrize`` passes to the ``model_path`` fixture.
MODEL_PATHS = {
    "test": TEST_MODEL_PATH,
    "post82": POST82_TEST_MODEL_PATH,
    "dense_lkj": DENSE_LKJ_TEST_MODEL_PATH,
    "rbc_second_order": RBC_SECOND_ORDER_TEST_MODEL_PATH,
    "rbc_multishock": RBC_MULTISHOCK_TEST_MODEL_PATH,
    "rbc_occbin": RBC_OCCBIN_TEST_MODEL_PATH,
    "nk_zlb": NK_ZLB_TEST_MODEL_PATH,
}


#: More memory than any test plan sizes to.
ABUNDANT_MEMORY_BYTES = 1 << 50


@pytest.fixture(autouse=True)
def abundant_memory() -> Iterator[None]:
    """Keep the machine's own free memory out of every test in the suite.

    The Monte Carlo memory profiler is the sole `psutil` consumer, and it reads
    physical and swap availability live. Left alone, any test running a plan
    large enough to approach whatever the machine happens to have free either
    warns or raises, so a pass depends on ambient load rather than on the code
    under test. Tests that care about scarcity pin their own readings.
    """
    import SymbolicDSGE.monte_carlo.memory as memory

    class _Reading:
        def __init__(self, **fields: int) -> None:
            self.__dict__.update(fields)

    virtual_memory = memory.psutil.virtual_memory
    swap_memory = memory.psutil.swap_memory
    memory.psutil.virtual_memory = lambda: _Reading(available=ABUNDANT_MEMORY_BYTES)
    memory.psutil.swap_memory = lambda: _Reading(free=ABUNDANT_MEMORY_BYTES)
    try:
        yield
    finally:
        memory.psutil.virtual_memory = virtual_memory
        memory.psutil.swap_memory = swap_memory


@pytest.fixture(scope="session")
def test_model_path() -> Path:
    return TEST_MODEL_PATH


@pytest.fixture(scope="session")
def test_model_yaml() -> str:
    """The text of the reference model, for the ``from_string`` entry points."""
    return TEST_MODEL_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def post82_test_model_path() -> Path:
    return POST82_TEST_MODEL_PATH


@pytest.fixture(scope="session")
def dense_lkj_test_model_path() -> Path:
    return DENSE_LKJ_TEST_MODEL_PATH


@pytest.fixture(scope="session")
def rbc_second_order_test_model_path() -> Path:
    return RBC_SECOND_ORDER_TEST_MODEL_PATH


@pytest.fixture(scope="session")
def rbc_multishock_test_model_path() -> Path:
    return RBC_MULTISHOCK_TEST_MODEL_PATH


@pytest.fixture(scope="session")
def rbc_occbin_test_model_path() -> Path:
    return RBC_OCCBIN_TEST_MODEL_PATH


@pytest.fixture(scope="session")
def nk_zlb_test_model_path() -> Path:
    return NK_ZLB_TEST_MODEL_PATH


@pytest.fixture(scope="session")
def gali_2015_test_model_path() -> Path:
    return GALI_2015_TEST_MODEL_PATH


@pytest.fixture
def model_path(request) -> Path:
    """One model per case of an indirect ``parametrize`` over ``MODEL_PATHS``.

    A decorator is evaluated at import time and cannot read a fixture, so the
    cases carry a name and the lookup happens here.
    """
    return MODEL_PATHS[request.param]


@pytest.fixture(scope="session")
def post82_bundle(post82_test_model_path):
    from SymbolicDSGE import DSGESolver, parse_config

    parsed = parse_config(post82_test_model_path)
    solver = DSGESolver(parsed)
    compiled = solver.compile()
    solved = solver.solve(compiled=compiled)
    return {
        "solver": solver,
        "parsed": parsed,
        "compiled": compiled,
        "solved": solved,
    }


@pytest.fixture(scope="session")
def solver_post82(post82_bundle):
    return post82_bundle["solver"]


@pytest.fixture(scope="session")
def parsed_post82(post82_bundle):
    return post82_bundle["parsed"]


@pytest.fixture(scope="session")
def compiled_post82(post82_bundle):
    return post82_bundle["compiled"]


@pytest.fixture(scope="session")
def solved_post82(post82_bundle):
    return post82_bundle["solved"]


@pytest.fixture(scope="session")
def test_bundle(test_model_path):
    from SymbolicDSGE import DSGESolver, parse_config

    parsed = parse_config(test_model_path)
    solver = DSGESolver(parsed)
    compiled = solver.compile()
    solved = solver.solve(compiled=compiled)
    return {
        "solver": solver,
        "parsed": parsed,
        "compiled": compiled,
        "solved": solved,
    }


@pytest.fixture(scope="session")
def solver_test(test_bundle):
    return test_bundle["solver"]


@pytest.fixture(scope="session")
def parsed_test(test_bundle):
    return test_bundle["parsed"]


@pytest.fixture(scope="session")
def compiled_test(test_bundle):
    return test_bundle["compiled"]


@pytest.fixture(scope="session")
def solved_test(test_bundle):
    """A solved model for the test fixture, for Monte Carlo tests."""
    return test_bundle["solved"]


@pytest.fixture(scope="session")
def rbc_so_bundle(rbc_second_order_test_model_path):
    from SymbolicDSGE import DSGESolver, parse_config

    parsed = parse_config(rbc_second_order_test_model_path)
    solver = DSGESolver(parsed)
    compiled = solver.compile()
    solved = solver.solve(compiled=compiled, order=2)
    return {
        "solver": solver,
        "parsed": parsed,
        "compiled": compiled,
        "solved": solved,
    }


@pytest.fixture(scope="session")
def solver_rbc_so(rbc_so_bundle):
    return rbc_so_bundle["solver"]


@pytest.fixture(scope="session")
def parsed_rbc_so(rbc_so_bundle):
    return rbc_so_bundle["parsed"]


@pytest.fixture(scope="session")
def compiled_rbc_so(rbc_so_bundle):
    return rbc_so_bundle["compiled"]


@pytest.fixture(scope="session")
def solved_rbc_so(rbc_so_bundle):
    return rbc_so_bundle["solved"]


@pytest.fixture(scope="session")
def solved_multishock(rbc_multishock_test_model_path):
    """The three-shock correlated RBC, solved to second order.

    Read-only: consumers take its policy and its calibration as they are, so
    the whole suite shares one solve. ``compiled`` comes off it.
    """
    from SymbolicDSGE import DSGESolver, parse_config

    model = parse_config(rbc_multishock_test_model_path)
    solver = DSGESolver(model)
    return solver.solve(solver.compile(), order=2)


@pytest.fixture(scope="session")
def y_post82(compiled_post82, solved_post82):
    """A simulated panel over POST82's observables, in their compiled order.

    The seed and ``T`` are fixed because the oracle goldens are values off this
    panel, not properties of it.
    """
    import numpy as np
    import pandas as pd

    calib = compiled_post82.config.calibration.parameters
    T = 48
    rng = np.random.default_rng(20260724)
    sim = solved_post82.sim(
        T=T,
        shocks={
            "e_g": rng.normal(0.0, calib["sig_g"], size=T),
            "e_z": rng.normal(0.0, calib["sig_z"], size=T),
            "e_r": rng.normal(0.0, calib["sig_r"], size=T),
        },
        x0=np.zeros((compiled_post82.n_var,), dtype=np.float64),
        observables=True,
    )
    return pd.DataFrame(
        {name: sim.observables[name] for name in compiled_post82.observable_names}
    )


@pytest.fixture
def post82_estimator(compiled_post82, solved_post82, y_post82):
    """Factory building an :class:`Estimator` on the shared POST82 model.

    Defaults to the two-parameter linear MLE setup the oracle uses; callers
    override ``mode`` / ``estimated_params`` as needed. A mapping passed as
    ``estimated_params`` carries the priors.
    """
    from SymbolicDSGE.estimation import Estimator

    def _make(mode="linear", estimated_params=("psi_pi", "rho_r")):
        return Estimator(
            compiled=compiled_post82,
            y=y_post82,
            observables=compiled_post82.observable_names,
            filter_mode=mode,
            estimated_params=estimated_params,
            ss_seed=solved_post82.policy.steady_state,
        )

    return _make
