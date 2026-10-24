# type: ignore
from __future__ import annotations

import copy

import pytest

from SymbolicDSGE.core.config import ModelConfig


@pytest.fixture
def conf(parsed_test: ModelConfig) -> ModelConfig:
    return copy.deepcopy(parsed_test)
