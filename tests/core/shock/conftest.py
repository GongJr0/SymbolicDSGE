"""Fixtures for shared shock tests."""

import pytest

from SymbolicDSGE import DSGESolver, ModelParser


@pytest.fixture(scope="session")
def solved_test_model(test_model_path):
    model, kalman = ModelParser(test_model_path).get_all()
    solver = DSGESolver(model, kalman)
    return solver.solve(solver.compile())
