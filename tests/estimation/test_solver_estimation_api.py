import numpy as np
import pandas as pd
import pytest

from SymbolicDSGE import DSGESolver
from SymbolicDSGE.estimation import Estimator, make_prior
from SymbolicDSGE.estimation.results import MLEResult


def test_solver_exposes_private_estimator_factory(
    solver_post82, compiled_post82, solved_post82, y_post82
):
    est = solver_post82._estimator(
        compiled=compiled_post82,
        y=y_post82,
        observables=compiled_post82.observable_names,
        estimated_params=["psi_pi"],
        ss_seed=solved_post82.policy.steady_state,
    )
    assert isinstance(est, Estimator)


def test_solver_estimate_validates_config_initial_guess_against_prior(
    solver_post82, compiled_post82, solved_post82, y_post82
):
    solver = solver_post82

    # The prior's transform is the parameter's domain, so map rejects an initial
    # guess outside it before the search: psi_pi is positive under the log
    # transform its gamma prior carries, and the config starts it at -1.0.
    with pytest.raises(ValueError, match="out of support"):
        solver.estimate(
            compiled=compiled_post82,
            y=y_post82,
            routine="map",
            observables=compiled_post82.observable_names,
            estimated_params={
                "psi_pi": make_prior("gamma", {"mean": 2.0, "std": 0.5}, "log")
            },
            ss_seed=solved_post82.policy.steady_state,
            theta0={"psi_pi": -1.0},
        )


def test_solver_estimate_and_solve_mle(post82_restore, y_post82, monkeypatch):
    solver = post82_restore["solver"]
    compiled = post82_restore["compiled"]

    # Spied rather than replaced: the solve has to run for ``solved`` to be a
    # real model, and the only thing a run cannot show is a kwarg that is never
    # passed.
    seen = {}
    real_solve = solver.solve

    def spy(*args, **kwargs):
        seen["parameters"] = kwargs.get("parameters")
        return real_solve(*args, **kwargs)

    monkeypatch.setattr(solver, "solve", spy)

    result, solved = solver.estimate_and_solve(
        compiled=compiled,
        y=y_post82,
        routine="mle",
        observables=compiled.observable_names,
        estimated_params=["psi_pi", "rho_r"],
        ss_seed=post82_restore["solved"].policy.steady_state,
        theta0={"psi_pi": 2.0, "rho_r": 0.8},
        bounds={"psi_pi": (1.0, 5.0), "rho_r": (0.0, 0.99)},
    )
    assert isinstance(result, MLEResult)
    # The estimated point reaches solve through the calibration, not a kwarg,
    # and the result names only what was estimated.
    assert seen["parameters"] is None
    assert set(result.theta) == {"psi_pi", "rho_r"}
    # No order asked for, and the linear filter's minimum is first.
    assert solved.policy.order == 1

    calibrated = _calibrated(solved)
    for name, value in result.theta.items():
        assert calibrated[name] == pytest.approx(float(value))


def test_solver_estimate_accepts_theta0_dictionary(
    solver_post82, compiled_post82, solved_post82, y_post82
):
    solver = solver_post82

    out = solver.estimate(
        compiled=compiled_post82,
        y=y_post82,
        routine="mle",
        observables=compiled_post82.observable_names,
        estimated_params=["psi_pi", "rho_r"],
        ss_seed=solved_post82.policy.steady_state,
        theta0={"psi_pi": 2.0, "rho_r": 0.8},
        bounds={"psi_pi": (1.0, 5.0), "rho_r": (0.0, 0.99)},
    )
    assert isinstance(out, MLEResult)
    assert out.success
    assert "psi_pi" in out.theta and "rho_r" in out.theta


def test_solver_estimate_rejects_theta0_naming_an_unestimated_parameter(
    solver_post82, compiled_post82, solved_post82, y_post82
):
    """A sparse theta0 fills from the calibration, so only a name that is not an
    estimation target is an error."""
    with pytest.raises(ValueError, match="not estimated"):
        solver_post82.estimate(
            compiled=compiled_post82,
            y=y_post82,
            routine="mle",
            observables=compiled_post82.observable_names,
            estimated_params=["psi_pi"],
            ss_seed=solved_post82.policy.steady_state,
            theta0={"rho_r": 0.8},
        )


def test_solver_theta0_dictionary_is_mapped_to_unconstrained_for_transformed_prior(
    solver_post82,
    compiled_post82,
    solved_post82,
    y_post82,
):
    solver = solver_post82

    # A log-transformed prior means the dict theta0 (given in parameter space) is
    # mapped through the transform to the unconstrained theta the driver expects;
    # a positive psi_pi must land in a finite basin rather than blowing up.
    prior = make_prior(
        distribution="log_normal",
        parameters={"mean": 0.0, "std": 0.5},
        transform="log",
    )
    with pytest.warns(UserWarning, match="MLE will ignore"):
        out = solver.estimate(
            compiled=compiled_post82,
            y=y_post82,
            routine="mle",
            observables=compiled_post82.observable_names,
            estimated_params={"psi_pi": prior},
            ss_seed=solved_post82.policy.steady_state,
            theta0={"psi_pi": 2.0},
            bounds={"psi_pi": (1.0, 5.0)},
        )
    assert out.success


def _calibrated(solved):
    """The solved model's parameter point, by name.

    ``solved.config`` is the compiled model's own config, so this is the
    calibration ``estimate_and_solve`` wrote the estimated point into.
    """
    return {
        str(name): float(value)
        for name, value in solved.config.calibration.parameters.items()
    }


def _normal_prior(mean, std):
    return make_prior(
        distribution="normal",
        parameters={"mean": mean, "std": std},
        transform="identity",
    )


def test_solver_estimate_and_solve_mcmc(post82_restore, y_post82):
    compiled = post82_restore["compiled"]
    result, solved = post82_restore["solver"].estimate_and_solve(
        compiled=compiled,
        y=y_post82,
        routine="mcmc",
        observables=compiled.observable_names,
        estimated_params={"psi_pi": _normal_prior(2.0, 0.5)},
        ss_seed=post82_restore["solved"].policy.steady_state,
        posterior_point="map",
        n_draws=30,
        burn_in=30,
        random_state=123,
    )

    assert result.samples.shape == (30, 1)
    # ``posterior_point="map"`` is the draw with the highest log-posterior, and
    # that draw is what the solved model was calibrated to.
    at_map = float(result.samples[int(np.argmax(result.logpost_trace)), 0])
    assert _calibrated(solved)["psi_pi"] == pytest.approx(at_map)


def test_solver_estimate_and_solve_mcmc_preserves_non_estimated_params(
    post82_restore, y_post82
):
    solver = post82_restore["solver"]
    compiled = post82_restore["compiled"]
    entered = _calibrated(solver.solve(compiled))["rho_r"]

    result, solved = solver.estimate_and_solve(
        compiled=compiled,
        y=y_post82,
        routine="mcmc",
        observables=compiled.observable_names,
        estimated_params={"psi_pi": _normal_prior(2.0, 0.5)},
        ss_seed=post82_restore["solved"].policy.steady_state,
        posterior_point="mean",
        n_draws=20,
        burn_in=20,
        random_state=123,
    )

    assert result.samples.shape == (20, 1)
    calibrated = _calibrated(solved)
    assert calibrated["psi_pi"] == pytest.approx(float(result.samples.mean(axis=0)[0]))
    # Only the estimated name is written back; the rest still read at the value
    # they entered with, captured before the run rather than after it.
    assert calibrated["rho_r"] == pytest.approx(entered)


def test_solver_estimate_and_solve_returns_the_order_it_was_asked_for(parsed_rbc_so):
    """Order is a statement about the returned model, not about the objective: a
    linear filter reads nothing above first order, so a second-order model is a
    valid thing to ask back from one.

    Second-order RBC is the smallest model an ``order=2`` return can come off:
    one shock, one observable, and a levels specification the expansion is taken
    about. Compiled here rather than shared, since the estimate writes its point
    into the compiled model's own calibration.
    """
    solver = DSGESolver(*parsed_rbc_so)
    compiled = solver.compile()

    # The config's own ss_seed resolves Newton here, and x0 defaults to the
    # steady state.
    reference = solver.solve(compiled=compiled, order=2)
    T = 40
    rng = np.random.default_rng(20261012)
    sim = reference.sim(
        T=T,
        shocks={"e": rng.normal(0.0, 0.01, size=T)},
        observables=True,
    )

    _, solved = solver.estimate_and_solve(
        compiled=compiled,
        y=pd.DataFrame({"c_obs": sim.observables["c_obs"]}),
        order=2,
        routine="mle",
        observables=["c_obs"],
        estimated_params=["rho"],
        theta0={"rho": 0.9},
        bounds={"rho": (0.0, 0.99)},
    )
    assert solved.policy.order == 2
