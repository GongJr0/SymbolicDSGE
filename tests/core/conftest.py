# type: ignore
from __future__ import annotations

import copy

import pytest

from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.config import ModelConfig
from SymbolicDSGE.core.model_parser import ParsedConfig
from SymbolicDSGE.core.solved_model import SolvedModel


@pytest.fixture(scope="module")
def parsed_test(test_model_path) -> ParsedConfig:
    return ModelParser(test_model_path).get_all()


@pytest.fixture
def conf(parsed_test: ParsedConfig) -> ModelConfig:
    return copy.deepcopy(parsed_test.model)


@pytest.fixture(scope="module")
def solved_rbc(rbc_second_order_test_model_path) -> SolvedModel:
    """A first-order solve of a model whose steady state is far from zero.

    ``solved_test`` and ``solved_post82`` both sit at a zero ``ss``.
    Above fixtures cannot test behavir dependent on linearization around a non-zero fixed point.
    KF/EKF parity is the specific case this fixture is intended to test.
    """
    model, kalman = ModelParser(rbc_second_order_test_model_path).get_all()
    solver = DSGESolver(model, kalman)
    return solver.solve(solver.compile(), order=1)
