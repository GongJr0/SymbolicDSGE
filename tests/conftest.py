from __future__ import annotations

from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
POST82_TEST_MODEL_PATH = TESTS_DIR / "fixtures" / "models" / "POST82.yaml"
DENSE_LKJ_TEST_MODEL_PATH = TESTS_DIR / "fixtures" / "models" / "LKJ_DENSE.yaml"
RBC_SECOND_ORDER_TEST_MODEL_PATH = (
    TESTS_DIR / "fixtures" / "models" / "rbc_second_order.yaml"
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # Pin every `sr`-marked test to a single xdist group so the pysr / Julia
    # fits all land on one worker under `--dist loadgroup`. Running them in
    # parallel makes multiple workers trigger juliapkg install + Julia
    # precompilation into the same depot at once, which deadlocks on the
    # precompile lock on a cold CI depot (green locally only because the depot
    # is already warm). Serializing them removes that race.
    for item in items:
        if item.get_closest_marker("sr") is not None:
            item.add_marker(pytest.mark.xdist_group("sr"))


#: More memory than any test plan sizes to.
ABUNDANT_MEMORY_BYTES = 1 << 50


@pytest.fixture(autouse=True)
def abundant_memory(monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr(
        memory.psutil,
        "virtual_memory",
        lambda: _Reading(available=ABUNDANT_MEMORY_BYTES),
    )
    monkeypatch.setattr(
        memory.psutil, "swap_memory", lambda: _Reading(free=ABUNDANT_MEMORY_BYTES)
    )


@pytest.fixture(scope="session")
def post82_test_model_path() -> Path:
    return POST82_TEST_MODEL_PATH


@pytest.fixture(scope="session")
def dense_lkj_test_model_path() -> Path:
    return DENSE_LKJ_TEST_MODEL_PATH


@pytest.fixture(scope="session")
def rbc_second_order_test_model_path() -> Path:
    return RBC_SECOND_ORDER_TEST_MODEL_PATH
