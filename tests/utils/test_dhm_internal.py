# type: ignore
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp
from numba import njit

from SymbolicDSGE import DSGESolver, ModelParser
import SymbolicDSGE.utils.dhm as dhm_module
from SymbolicDSGE.utils.dhm import DenHaanMarcet


@pytest.fixture(scope="module")
def solved_test():
    model, kalman = ModelParser("MODELS/test.yaml").get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile(n_state=3, n_exog=2)
    return solver.solve(compiled)


@njit
def _objective_for_moments(fwd, cur, params):
    return np.array(
        [
            fwd[0] - cur[0] + params[0],
            (fwd[1] - cur[1]) + 0.0j,
        ],
        dtype=np.complex128,
    )


@njit
def _foc_for_moments(fwd, cur, params):
    return np.array(
        [
            fwd[0] + cur[0] + params[0],
            cur[1] - fwd[1],
        ],
        dtype=np.float64,
    )


def test_low_level_state_and_moment_builders_match_manual_construction():
    A = np.array([[0.5, 0.0], [0.0, 1.0]], dtype=np.float64)
    B = np.array([[1.0], [0.5]], dtype=np.float64)
    x0 = np.array([1.0, 2.0], dtype=np.float64)
    shock_mat = np.array([[0.5], [-1.0]], dtype=np.float64)
    states = dhm_module._simulate_linear_states(A, B, x0, shock_mat)
    expected_states = np.array(
        [
            [1.0, 2.0],
            [1.0, 2.25],
            [-0.5, 1.75],
        ],
        dtype=np.float64,
    )
    assert np.allclose(states, expected_states)

    current_states = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    forward_states = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    params = np.array([1.0], dtype=np.complex128)

    moments, residuals, instruments = dhm_module._build_forward_moments(
        current_states,
        forward_states,
        params,
        _objective_for_moments,
        np.array([0, 1], dtype=np.int64),
        np.array([1], dtype=np.int64),
        True,
        0,
    )
    assert np.allclose(
        residuals,
        np.array([[5.0, 4.0], [5.0, 4.0]], dtype=np.float64),
    )
    assert np.allclose(
        instruments,
        np.array([[1.0, 2.0], [1.0, 4.0]], dtype=np.float64),
    )
    assert np.allclose(
        moments,
        np.array(
            [
                [5.0, 10.0, 4.0, 8.0],
                [5.0, 20.0, 4.0, 16.0],
            ],
            dtype=np.float64,
        ),
    )

    lagged_moments, lagged_residuals, lagged_instruments = (
        dhm_module._build_lagged_foc_moments(
            current_states,
            forward_states,
            np.array([0.0], dtype=np.float64),
            _foc_for_moments,
            np.array([0], dtype=np.int64),
            False,
            0,
        )
    )
    assert np.allclose(
        lagged_residuals,
        np.array([[6.0, -4.0], [10.0, -4.0]], dtype=np.float64),
    )
    assert np.allclose(
        lagged_instruments,
        np.array([[1.0], [3.0]], dtype=np.float64),
    )
    assert np.allclose(
        lagged_moments,
        np.array([[6.0, -4.0], [30.0, -12.0]], dtype=np.float64),
    )


def test_dhm_index_and_observable_validators_cover_default_and_error_paths(solved_test):
    dhm = DenHaanMarcet(solved_test)

    assert np.array_equal(
        dhm._resolve_equation_idx(None),
        np.arange(len(solved_test.compiled.objective_eqs), dtype=np.int64),
    )
    with pytest.raises(ValueError, match="At least one equation index"):
        dhm._resolve_equation_idx([])
    with pytest.raises(IndexError, match="must lie in"):
        dhm._resolve_equation_idx([99])
    with pytest.raises(ValueError, match="must be unique"):
        dhm._resolve_equation_idx([0, 0])

    assert np.array_equal(
        dhm._resolve_instrument_idx(None, True),
        np.arange(len(solved_test.compiled.var_names), dtype=np.int64),
    )
    with pytest.raises(KeyError, match="Unknown instrument variable"):
        dhm._resolve_instrument_idx(["ghost"], True)
    with pytest.raises(ValueError, match="At least one instrument index"):
        dhm._resolve_instrument_idx([], False)
    with pytest.raises(IndexError, match="must lie in"):
        dhm._resolve_instrument_idx([-1], True)
    with pytest.raises(ValueError, match="must be unique"):
        dhm._resolve_instrument_idx([0, 0], True)

    with pytest.raises(ValueError, match="At least one observable"):
        dhm._requested_observable_names([])
    with pytest.raises(ValueError, match="duplicates"):
        dhm._resolve_observable_names(["Infl", "Infl"])
    with pytest.raises(KeyError, match="Unknown observables"):
        dhm._resolve_observable_names(["ghost"])


def test_prepare_measurement_observed_covers_permutation_and_input_errors(solved_test):
    dhm = DenHaanMarcet(solved_test)
    requested = ("Rate", "Infl")
    resolved = ("Infl", "Rate")
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    permuted = dhm._prepare_measurement_observed(arr, requested, resolved)
    assert np.allclose(permuted, np.array([[2.0, 1.0], [4.0, 3.0]], dtype=np.float64))

    mapped = dhm._prepare_measurement_observed(
        {"Infl": np.array([2.0, 4.0]), "Rate": np.array([1.0, 3.0])},
        requested,
        resolved,
    )
    assert np.allclose(mapped, permuted)

    with pytest.raises(KeyError, match="missing observables"):
        dhm._prepare_measurement_observed(
            {"Infl": np.array([1.0, 2.0])}, requested, resolved
        )
    with pytest.raises(ValueError, match="share the same length"):
        dhm._prepare_measurement_observed(
            {"Infl": np.array([1.0, 2.0]), "Rate": np.array([1.0])},
            requested,
            resolved,
        )
    with pytest.raises(ValueError, match="2D array or mapping"):
        dhm._prepare_measurement_observed(np.array([1.0, 2.0]), requested, resolved)
    with pytest.raises(ValueError, match="1D/2D array or mapping"):
        dhm._prepare_measurement_observed(
            np.zeros((2, 2, 1), dtype=np.float64), requested, resolved
        )
    with pytest.raises(ValueError, match="columns"):
        dhm._prepare_measurement_observed(
            np.zeros((2, 3), dtype=np.float64), requested, resolved
        )


def test_dhm_foc_internal_validators_cover_remaining_error_paths(solved_test):
    dhm = DenHaanMarcet(solved_test)
    local_dict, var_funcs, param_syms, shock_syms, _ = dhm._foc_parse_context(None)

    with pytest.raises(ValueError, match="At least one FOC expression"):
        dhm._parse_and_normalize_focs((), None)

    with pytest.raises(ValueError, match="non-empty"):
        dhm._parse_foc_locals(
            dict(local_dict),
            var_funcs,
            param_syms,
            shock_syms,
            {"   ": "1.0"},
        )

    with pytest.raises(ValueError, match="Use either 'name' or 'name\\(t\\)'"):
        dhm._parse_foc_locals(
            dict(local_dict),
            var_funcs,
            param_syms,
            shock_syms,
            {"bad(t+1)": "1.0"},
        )

    with pytest.raises(ValueError, match="exactly one '='"):
        dhm._parse_foc_text("x(t) = Pi(t) = 0", local_dict, tuple(var_funcs))

    with pytest.raises(ValueError, match="explicit time index"):
        dhm._parse_foc_text("x", local_dict, tuple(var_funcs))

    with pytest.raises(ValueError, match="must reference at least one time-indexed"):
        dhm._validate_expression_symbols(
            sp.Integer(1),
            var_funcs,
            param_syms,
            shock_syms,
            require_time_vars=True,
        )

    with pytest.raises(KeyError, match="Unknown time-dependent symbol"):
        dhm._validate_expression_symbols(
            sp.Function("ghost")(dhm._t),
            var_funcs,
            param_syms,
            shock_syms,
            require_time_vars=False,
        )

    with pytest.raises(ValueError, match="single time argument"):
        dhm._validate_expression_symbols(
            var_funcs["x"](dhm._t, dhm._t),
            var_funcs,
            param_syms,
            shock_syms,
            require_time_vars=False,
        )

    with pytest.raises(ValueError, match="integer time offsets"):
        dhm._validate_expression_symbols(
            var_funcs["x"](dhm._t + sp.Symbol("s")),
            var_funcs,
            param_syms,
            shock_syms,
            require_time_vars=False,
        )

    with pytest.raises(ValueError, match="span at most one period"):
        dhm._validate_expression_symbols(
            var_funcs["x"](dhm._t - 1) + var_funcs["Pi"](dhm._t + 1),
            var_funcs,
            param_syms,
            shock_syms,
            require_time_vars=False,
        )

    with pytest.raises(ValueError, match="Shock innovations are not supported"):
        dhm._validate_expression_symbols(
            next(iter(shock_syms.values())),
            var_funcs,
            param_syms,
            shock_syms,
            require_time_vars=False,
        )

    with pytest.raises(KeyError, match="Unknown symbol 'mystery'"):
        dhm._validate_expression_symbols(
            sp.Symbol("mystery"),
            var_funcs,
            param_syms,
            shock_syms,
            require_time_vars=False,
        )

    with pytest.raises(ValueError, match="must be called with a single time argument"):
        dhm._expand_foc_locals(
            sp.Function("Alias")(dhm._t, dhm._t),
            {
                "Alias": dhm_module._FocLocalDef(
                    kind="function",
                    symbol=sp.Function("Alias"),
                    expr=var_funcs["x"](dhm._t),
                )
            },
        )

    with pytest.raises(ValueError, match="Failed to normalize all model variables"):
        dhm._build_foc_vector_func([sp.Function("ghost")(dhm._t)])

    with pytest.raises(ValueError, match="Unresolved symbols remain"):
        dhm._build_foc_vector_func([sp.Symbol("mystery")])
