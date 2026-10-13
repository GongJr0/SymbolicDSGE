# type: ignore
from __future__ import annotations

import copy

import pytest

from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.config import ModelConfig
from SymbolicDSGE.core.model_parser import ParsedConfig
from SymbolicDSGE.core.solved_model import SolvedModel


@pytest.fixture
def conf(parsed_test: ParsedConfig) -> ModelConfig:
    return copy.deepcopy(parsed_test.model)


@pytest.fixture(scope="session")
def solved_multishock(rbc_multishock_test_model_path) -> SolvedModel:
    """The three-shock correlated RBC, solved to second order.

    Read-only: the parity tests take its policy and its calibration as they
    are, so the whole package shares one solve. ``compiled`` comes off it.
    """
    model, kalman = ModelParser(rbc_multishock_test_model_path).get_all()
    solver = DSGESolver(model, kalman)
    return solver.solve(solver.compile(), order=2)
