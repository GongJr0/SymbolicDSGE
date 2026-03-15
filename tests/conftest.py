from __future__ import annotations

from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
POST82_TEST_MODEL_PATH = TESTS_DIR / "fixtures" / "models" / "POST82.yaml"


@pytest.fixture(scope="session")
def post82_test_model_path() -> Path:
    return POST82_TEST_MODEL_PATH
