# type: ignore
from __future__ import annotations

import pytest

from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.compiled_model import CompiledModel
from SymbolicDSGE.core.model_parser import ParsedConfig
from SymbolicDSGE.core.solved_model import SolvedModel


@pytest.fixture(scope="module")
def parsed_test() -> ParsedConfig:
    return ModelParser("MODELS/test.yaml").get_all()


@pytest.fixture(scope="module")
def parsed_post82() -> ParsedConfig:
    return ModelParser("MODELS/POST82.yaml").get_all()


@pytest.fixture(scope="module")
def solver_test(parsed_test: ParsedConfig) -> DSGESolver:
    model, kalman = parsed_test
    return DSGESolver(model, kalman)


@pytest.fixture(scope="module")
def solver_post82(parsed_post82: ParsedConfig) -> DSGESolver:
    model, kalman = parsed_post82
    return DSGESolver(model, kalman)


@pytest.fixture(scope="module")
def compiled_test(solver_test: DSGESolver) -> CompiledModel:
    return solver_test.compile(n_state=3, n_exog=2)


@pytest.fixture(scope="module")
def compiled_post82(solver_post82: DSGESolver) -> CompiledModel:
    return solver_post82.compile(n_state=3, n_exog=3)


@pytest.fixture(scope="module")
def solved_test(solver_test: DSGESolver, compiled_test: CompiledModel) -> SolvedModel:
    return solver_test.solve(compiled_test)


@pytest.fixture(scope="module")
def solved_post82(
    solver_post82: DSGESolver, compiled_post82: CompiledModel
) -> SolvedModel:
    return solver_post82.solve(compiled_post82)
