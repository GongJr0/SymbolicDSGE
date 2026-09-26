"""Fixtures for shared shock tests."""

import pytest

from SymbolicDSGE import DSGESolver, ModelParser


@pytest.fixture(scope="session")
def solved_test_model():
    model, kalman = ModelParser("MODELS/test.yaml").get_all()
    solver = DSGESolver(model, kalman)
    return solver.solve(solver.compile())
