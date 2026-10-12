# type: ignore
from __future__ import annotations

import copy

import pytest

from SymbolicDSGE.core.config import ModelConfig
from SymbolicDSGE.core.model_parser import ParsedConfig


@pytest.fixture
def conf(parsed_test: ParsedConfig) -> ModelConfig:
    return copy.deepcopy(parsed_test.model)
