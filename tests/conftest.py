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


@pytest.fixture
def model_path(request) -> Path:
    """One model per case of an indirect ``parametrize`` over ``MODEL_PATHS``.

    A decorator is evaluated at import time and cannot read a fixture, so the
    cases carry a name and the lookup happens here.
    """
    return MODEL_PATHS[request.param]


@pytest.fixture(scope="session")
def solved_rbc_second_order(rbc_second_order_test_model_path):
    """Second-order RBC model with positive observation noise for filter tests."""
    import numpy as np
    from SymbolicDSGE import DSGESolver, ModelParser
    from SymbolicDSGE.kalman.config import KalmanConfig

    model, _ = ModelParser(rbc_second_order_test_model_path).get_all()
    solver = DSGESolver(model, KalmanConfig(R=np.array([[0.01]], dtype=np.float64)))
    return solver.solve(solver.compile(), order=2)
