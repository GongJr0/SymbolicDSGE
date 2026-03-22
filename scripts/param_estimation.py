# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: SymbolicDSGE
#     language: python
#     name: symbolicdsge
# ---

# %%
from SymbolicDSGE import ModelParser, DSGESolver, Shock
from SymbolicDSGE.utils import FRED
from SymbolicDSGE.utils.math_utils import HP_two_sided, annualized_log_percent
from SymbolicDSGE.bayesian import make_prior
from SymbolicDSGE.regression import TemplateConfig, PySRParams

from sympy import Matrix, Float, preorder_traversal
from warnings import catch_warnings, simplefilter

from numpy import log
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import cProfile

_FIGSIZE_1D = (10, 6)
_FIGSIZE_2D = (12, 6)
# %%
parser = ModelParser("../MODELS/classes/reference.yaml")
model, kalman = parser.get_all()

with catch_warnings():
    # Equations in a sp.Matrix are deprecated, this is only used as a pretty print function
    simplefilter(action="ignore")
    mat = Matrix(model.equations.model)
mat
# %%
fred = FRED(
    key_env=None,  # None => look for the ".env" file. If you have a custom env file, provide its path here.
    key_name="FRED_KEY",  # Name of the variable in the env file that contains the FRED API key.
)
df = fred.get_frame(
    series_ids=[
        "A939RX0Q048SBEA",  # Real GDP per Cap.
        "CPIAUCSL",  # Consumer Price Index for All Urban Consumers: All Items
        "FEDFUNDS",  # Effective Federal Funds Rate
    ],
    date_range=(
        "1955-01-01",
        "2007-10-01",
    ),  # Date range for the data ("YYYY-MM-DD" format or a pd.DatetimeIndex object)
)

gdp_q = df["A939RX0Q048SBEA"]  # already quarterly in most pulls; verify freq

cpi_q = df["CPIAUCSL"].resample("QS").mean()  # quarterly avg CPI
ffr_q = df["FEDFUNDS"].resample("QS").mean()  # quarterly avg policy rate

idx_range = pd.date_range(start="1984-01-01", end="2007-01-01", freq="QS")


df = pd.DataFrame(
    {
        "A939RX0Q048SBEA": gdp_q.reindex(idx_range),
        "CPIAUCSL": cpi_q.reindex(idx_range),
        "FEDFUNDS": ffr_q.reindex(idx_range),
    }
)

df

# %%
x_trend = HP_two_sided(log(df["A939RX0Q048SBEA"]), lamb=1600)[
    0
]  # returns (trend, cycle)
x = (
    log(df["A939RX0Q048SBEA"]) - x_trend
) * 100  # HP detrended quarterly log output gap


inf_lvl = annualized_log_percent(df["CPIAUCSL"], periods_per_year=4)
rate_lvl = df["FEDFUNDS"]

r_ss = model.calibration.parameters["r_star"]
pi_ss = model.calibration.parameters["pi_star"]

rate = (rate_lvl - (r_ss + pi_ss)) / 4  # gap to steady state
inf = (inf_lvl - pi_ss) / 4  # gap to steady state

df_model_units = pd.DataFrame(
    {
        "r": rate,
        "Pi": inf,
        "x": x,
    }
).dropna()

observed = pd.DataFrame(
    {
        "OutGap": df_model_units["x"],
        "Infl": inf_lvl[df_model_units.index],
        "Rate": rate_lvl[df_model_units.index],
    }
)
observed.index = df_model_units.index

# %%


def seed_increment():
    """Counter for random seeds"""
    n = -1
    while True:
        n += 1
        yield n


seed = seed_increment()

prior_spec = {
    # (0, 1)
    "beta": make_prior(
        "beta",
        parameters={
            "a": 100 * 0.99,
            "b": 100 * 0.001,
            "random_state": next(seed),
        },
        transform="logit",
    ),
    "rho_r": make_prior(
        "beta",
        parameters={
            "a": 100 * 0.84,
            "b": 100 * 0.16,
            "random_state": next(seed),
        },
        transform="logit",
    ),
    "rho_g": make_prior(
        "beta",
        parameters={
            "a": 100 * 0.83,
            "b": 100 * 0.17,
            "random_state": next(seed),
        },
        transform="logit",
    ),
    "rho_z": make_prior(
        "beta",
        parameters={
            "a": 100 * 0.85,
            "b": 100 * 0.15,
            "random_state": next(seed),
        },
        transform="logit",
    ),
    # (0, +inf)
    "psi_pi": make_prior(
        "gamma",
        parameters={
            "mean": 2.19,
            "std": 0.5,
            "random_state": next(seed),
        },
        transform="log",
    ),
    "psi_x": make_prior(
        "gamma",
        parameters={
            "mean": 0.30,
            "std": 0.1,
            "random_state": next(seed),
        },
        transform="log",
    ),
    "kappa": make_prior(
        "gamma",
        parameters={
            "mean": 0.58,
            "std": 0.1,
            "random_state": next(seed),
        },
        transform="log",
    ),
    "tau_inv": make_prior(
        "gamma",
        parameters={
            "mean": 1.86,
            "std": 0.5,
            "random_state": next(seed),
        },
        transform="log",
    ),
    # Correlation (-1,1)
    "rho_gz": make_prior(
        "trunc_normal",
        parameters={
            "mean": 0.0,
            "std": 0.20,
            "low": -1.0,
            "high": 1.0,
            "random_state": next(seed),
        },
        transform="affine_logit",
        transform_kwargs={
            "low": -1.0,
            "high": 1.0,
        },
    ),
    # Shock std devs (0, +inf)
    "sig_r": make_prior(
        "gamma",
        parameters={
            "mean": 0.18,
            "std": 0.1,
            "random_state": next(seed),
        },
        transform="log",
    ),
    "sig_g": make_prior(
        "gamma",
        parameters={
            "mean": 0.18,
            "std": 0.1,
            "random_state": next(seed),
        },
        transform="log",
    ),
    "sig_z": make_prior(
        "gamma",
        parameters={
            "mean": 0.64,
            "std": 0.1,
            "random_state": next(seed),
        },
        transform="log",
    ),
    "meas_outgap": make_prior(
        "gamma",
        parameters={
            "mean": 0.5,
            "std": 0.1,
            "random_state": next(seed),
        },
        transform="log",
    ),
    "meas_infl": make_prior(
        "gamma",
        parameters={
            "mean": 0.5,
            "std": 0.1,
            "random_state": next(seed),
        },
        transform="log",
    ),
    "meas_rate": make_prior(
        "gamma",
        parameters={
            "mean": 0.5,
            "std": 0.1,
            "random_state": next(seed),
        },
        transform="log",
    ),
    "R_corr": make_prior(
        "lkj_chol",
        parameters={
            "eta": 1.0,
            "K": 3,
            "random_state": next(seed),
        },
        transform="cholesky_corr",
    ),
}

solver = DSGESolver(model, kalman)
comp = solver.compile(
    n_exog=3,
    n_state=3,
)
estim = lambda: solver.estimate_and_solve(
    compiled=comp,
    y=observed.loc[observed.index >= "1984-01-01", :],
    priors=prior_spec,
    method="mcmc",
    posterior_point="mean",
    steady_state=[0.0, 0.0, 0.0, 0.0, 0.0],
    estimated_params=list(prior_spec.keys()),
    n_draws=25_000,
    burn_in=10_000,
    thin=1,
    update_R_in_iterations=True,
)
res, sol = estim()
parser.update_calibration_parameters(
    sol.config,
    digits=3,
    output_path="../MODELS/classes/base.yaml",
)  # Make config file with new parameters
parser.update_calibration_parameters(
    sol.config,
    digits=3,
    output_path="../MODELS/classes/augmented.yaml",
)  # Duplicate parameters for augmented model

# cProfile.run("res, sol = estim()", sort="cumtime")

# %%
param_names = res.param_names

best_idx = np.argmax(res.logpost_trace)

post_mean = np.mean(res.samples, axis=0)
loglik = np.mean(res.logpost_trace)
accept_rate = res.accept_rate
n_draws = res.n_draws
burn_in = res.burn_in
thin = res.thin

param_to_val = dict(zip(param_names, post_mean))


pd.Series(
    {
        **param_to_val,
        "loglik": loglik,
        "accept_rate": accept_rate,
        "n_draws": n_draws,
        "burn_in": burn_in,
        "thin": thin,
    }
)

# %%
sol.transition_plot(shocks=["g"], T=25, scale=1, observables=True)
# %%
kf = sol.kalman(
    observed.loc[observed.index >= "1984-01-01", :],
    "linear",
    return_shocks=True,
)
# %%
obs = observed.loc[observed.index >= "1984-01-01", :]
idx = obs.index

fig, ax = plt.subplots(3, 1, figsize=_FIGSIZE_1D)

plt.suptitle("Filtered, Predicted vs Actual Measurements")

ax[0].plot(idx, kf.y_pred[:, 0], label="Predicted")
ax[0].plot(idx, kf.y_filt[:, 0], label="Filtered")
ax[0].plot(idx, obs.iloc[:, 0], label="Actual", linestyle="--", alpha=0.7)
ax[0].set_title("Output Gap")
ax[0].legend()

ax[1].plot(idx, kf.y_pred[:, 1], label="Predicted")
ax[1].plot(idx, kf.y_filt[:, 1], label="Filtered")
ax[1].plot(idx, obs.iloc[:, 1], label="Actual", linestyle="--", alpha=0.7)
ax[1].set_title("Inflation")
ax[1].legend()

ax[2].plot(idx, kf.y_pred[:, 2], label="Predicted")
ax[2].plot(idx, kf.y_filt[:, 2], label="Filtered")
ax[2].plot(idx, obs.iloc[:, 2], label="Actual", linestyle="--", alpha=0.7)
ax[2].set_title("Policy Rate")
ax[2].legend()

plt.tight_layout()
# %%
gz = Shock(T=len(idx) - 1, dist="norm", multivar=True, seed=0).shock_generator()
r = Shock(T=len(idx) - 1, dist="norm", seed=1).shock_generator()
sim = sol.sim(
    T=len(idx) - 1,
    shocks={"g,z": gz, "r": r},
    observables=True,
)
# %%
fig, ax = plt.subplots(3, 1, figsize=_FIGSIZE_1D)

plt.suptitle("Stochastic Simulation vs Actual")

ax[0].plot(idx, sim["OutGap"], label="Simulated")
ax[0].plot(idx, obs.iloc[:, 0], label="Actual", linestyle="--", alpha=0.7)
ax[0].set_title("Output Gap")
ax[0].legend()

ax[1].plot(idx, sim["Infl"], label="Simulated")
ax[1].plot(idx, obs.iloc[:, 1], label="Actual", linestyle="--", alpha=0.7)
ax[1].set_title("Inflation")
ax[1].legend()

ax[2].plot(idx, sim["Rate"], label="Simulated")
ax[2].plot(idx, obs.iloc[:, 2], label="Actual", linestyle="--", alpha=0.7)
ax[2].set_title("Policy Rate")
ax[2].legend()

plt.tight_layout()
# %%
fig, ax = plt.subplots(3, 2, figsize=_FIGSIZE_2D)

plt.suptitle("Innovation AutoCorr")
plot_acf(
    kf.innov[:, 0],
    ax=ax[0, 0],
    lags=20,
    title="Output Gap Innovation ACF",
)
plot_acf(
    kf.innov[:, 1],
    ax=ax[1, 0],
    lags=20,
    title="Inflation Innovation ACF",
)
plot_acf(
    kf.innov[:, 2],
    ax=ax[2, 0],
    lags=20,
    title="Policy Rate Innovation ACF",
)
plot_pacf(
    kf.innov[:, 0],
    ax=ax[0, 1],
    lags=20,
    title="Output Gap Innovation PACF",
)
plot_pacf(
    kf.innov[:, 1],
    ax=ax[1, 1],
    lags=20,
    title="Inflation Innovation PACF",
)
plot_pacf(
    kf.innov[:, 2],
    ax=ax[2, 1],
    lags=20,
    title="Policy Rate Innovation PACF",
)

plt.tight_layout()
# %%
template = TemplateConfig(
    include_expression=False,
    interaction_form="prod",
    hessian_restriction="free",
    power_law_lower_bound=2,
    power_law_upper_bound=2,
    powers_in_interactions=False,
    constant_filtering="parametrize_all",
    model_complexity_bound=20,
)

params = PySRParams(
    niterations=500,
    maxsize=20,
    complexity_of_constants=3,
    complexity_of_variables=1,
    deterministic=True,
    random_state=0,
    parallelism="serial",
)

sr_discovery = lambda obs: sol.fit_kf(
    template_config=template,
    sr_params=params,
    y=observed.loc[observed.index >= "1984-01-01", :],
    variables=["r", "Pi", "x"],
    observable=obs,
).expressions

# Only run sr for observalbes with innovation autocorrelation.
x_sr = sr_discovery("OutGap")
# r_sr = sr_discovery("Rate")
# pi_sr = sr_discovery("Infl")


# %%
def sum_moments(series):
    out = pd.Series()

    out["mean"] = series.mean()
    out["var"] = series.var()
    out["Q1"] = series.quantile(0.25)
    out["Median"] = series.median()
    out["Q3"] = series.quantile(0.75)
    out["IQR"] = out["Q3"] - out["Q1"]
    return out.round(2)


sum_moments(pd.Series(kf.y_filt[:, 1]))


# %%
def walk_round(x, n=3):
    for atom in preorder_traversal(x):
        if isinstance(atom, Float):
            x = x.subs(atom, round(atom, n))
    return x


x_sr["initial_expr"] = x_sr["initial_expr"].apply(lambda x: walk_round(x, n=3))
x_sr[["initial_expr", "sympy_format", "loss", "complexity"]]
# %%
# Augmented Model
parser_aug = ModelParser("../MODELS/augmented.yaml")
conf_aug, kalman_aug = parser_aug.get_all()

solver_aug = DSGESolver(conf_aug, kalman_aug)
comp_aug = solver_aug.compile(n_exog=3, n_state=3)

aug = solver_aug.solve(compiled=comp_aug)  # Solve without re-estimation
parser_aug.update_calibration_parameters(
    aug.config,
    digits=3,
    output_path="../MODELS/classes/augmented.yaml",
)  # Update augmented config to reflect current parameters (same as base for now, since we haven't re-estimated yet)
# %%
aug.transition_plot(shocks=["g"], T=25, scale=1, observables=True)

# %%
kf_aug = aug.kalman(
    observed.loc[observed.index >= "1984-01-01", :],
    "extended",
    return_shocks=True,
)

# %%
obs = observed.loc[observed.index >= "1984-01-01", :]
idx = obs.index

fig, ax = plt.subplots(3, 1, figsize=_FIGSIZE_1D)

plt.suptitle("Augmented Model: Filtered, Predicted vs Actual Measurements")

ax[0].plot(idx, kf_aug.y_pred[:, 0], label="Predicted")
ax[0].plot(idx, kf_aug.y_filt[:, 0], label="Filtered")
ax[0].plot(idx, obs.iloc[:, 0], label="Actual", linestyle="--", alpha=0.7)
ax[0].set_title("Output Gap")
ax[0].legend()

ax[1].plot(idx, kf_aug.y_pred[:, 1], label="Predicted")
ax[1].plot(idx, kf_aug.y_filt[:, 1], label="Filtered")
ax[1].plot(idx, obs.iloc[:, 1], label="Actual", linestyle="--", alpha=0.7)
ax[1].set_title("Inflation")
ax[1].legend()

ax[2].plot(idx, kf_aug.y_pred[:, 2], label="Predicted")
ax[2].plot(idx, kf_aug.y_filt[:, 2], label="Filtered")
ax[2].plot(idx, obs.iloc[:, 2], label="Actual", linestyle="--", alpha=0.7)
ax[2].set_title("Policy Rate")
ax[2].legend()

plt.tight_layout()
# %%
sim_aug = aug.sim(
    T=len(idx) - 1,
    shocks={"g,z": gz, "r": r},
    observables=True,
)
fig, ax = plt.subplots(3, 1, figsize=_FIGSIZE_1D)

plt.suptitle("Augmented Model: Stochastic Simulation vs Actual")

ax[0].plot(idx, sim_aug["OutGap"], label="Simulated")
ax[0].plot(idx, obs.iloc[:, 0], label="Actual", linestyle="--", alpha=0.7)
ax[0].set_title("OutGap")
ax[0].legend()

ax[1].plot(idx, sim_aug["Infl"], label="Simulated")
ax[1].plot(idx, obs.iloc[:, 1], label="Actual", linestyle="--", alpha=0.7)
ax[1].set_title("Inflation")
ax[1].legend()

ax[2].plot(idx, sim_aug["Rate"], label="Simulated")
ax[2].plot(idx, obs.iloc[:, 2], label="Actual", linestyle="--", alpha=0.7)
ax[2].set_title("Policy Rate")
ax[2].legend()

plt.tight_layout()

# %%
fig, ax = plt.subplots(3, 2, figsize=_FIGSIZE_2D)

plt.suptitle("Augmented Model: Innovation AutoCorr")

plot_acf(
    kf_aug.innov[:, 0],
    ax=ax[0, 0],
    lags=20,
    title="Output Gap Innovations ACF",
)
plot_acf(
    kf_aug.innov[:, 1],
    ax=ax[1, 0],
    lags=20,
    title="Inflation Innovations ACF",
)
plot_acf(
    kf_aug.innov[:, 2],
    ax=ax[2, 0],
    lags=20,
    title="Policy Rate Innovations ACF",
)
plot_pacf(
    kf_aug.innov[:, 0],
    ax=ax[0, 1],
    lags=20,
    title="Output Gap Innovations PACF",
)
plot_pacf(
    kf_aug.innov[:, 1],
    ax=ax[1, 1],
    lags=20,
    title="Inflation Innovations PACF",
)
plot_pacf(
    kf_aug.innov[:, 2],
    ax=ax[2, 1],
    lags=20,
    title="Policy Rate Innovations PACF",
)

plt.tight_layout()
# %%

sum_moments(pd.Series(sim_aug["Infl"]))

# %%
# Augmented + Re-estimated model
aug_priors = {
    **prior_spec,
    **{
        "pi_const": make_prior(
            "normal",
            parameters={"mean": -0.282, "std": 0.1},
            transform="identity",
        ),
    },
}

res_aug, sol_aug = solver_aug.estimate_and_solve(
    compiled=comp_aug,
    y=observed.loc[observed.index >= "1984-01-01", :],
    priors=aug_priors,
    method="mcmc",
    posterior_point="mean",
    steady_state=[0.0, 0.0, 0.0, 0.0, 0.0],
    estimated_params=list(aug_priors.keys()),
    n_draws=25_000,
    burn_in=10_000,
    thin=1,
)
parser_aug.update_calibration_parameters(
    sol_aug.config,
    digits=3,
    output_path="../MODELS/classes/augmented_reestimated.yaml",
)  # Store re-estimation results in a new config file
# %%
# Summarize results
param_names_aug = res_aug.param_names
post_mean_aug = np.mean(res_aug.samples, axis=0)
loglik_aug = np.mean(res_aug.logpost_trace)
accept_rate_aug = res_aug.accept_rate
pd.Series(
    {
        **dict(zip(param_names_aug, post_mean_aug)),
        "loglik": loglik_aug,
        "accept_rate": accept_rate_aug,
        "n_draws": res_aug.n_draws,
        "burn_in": res_aug.burn_in,
        "thin": res_aug.thin,
    }
)
# %%
sol_aug.transition_plot(shocks=["g"], T=25, scale=1, observables=True)

# %%
kf_aug_reest = sol_aug.kalman(
    observed.loc[observed.index >= "1984-01-01", :],
    "extended",
    return_shocks=True,
)
# %%
obs = observed.loc[observed.index >= "1984-01-01", :]
idx = obs.index

fig, ax = plt.subplots(3, 1, figsize=_FIGSIZE_1D)

plt.suptitle(
    "Augmented + Re-estimated Model: Filtered, Predicted vs Actual Measurements"
)

ax[0].plot(idx, kf_aug_reest.y_pred[:, 0], label="Predicted")
ax[0].plot(idx, kf_aug_reest.y_filt[:, 0], label="Filtered")
ax[0].plot(idx, obs.iloc[:, 0], label="Actual", linestyle="--", alpha=0.7)
ax[0].set_title("Output Gap")
ax[0].legend()

ax[1].plot(idx, kf_aug_reest.y_pred[:, 1], label="Predicted")
ax[1].plot(idx, kf_aug_reest.y_filt[:, 1], label="Filtered")
ax[1].plot(idx, obs.iloc[:, 1], label="Actual", linestyle="--", alpha=0.7)
ax[1].set_title("Inflation")
ax[1].legend()

ax[2].plot(idx, kf_aug_reest.y_pred[:, 2], label="Predicted")
ax[2].plot(idx, kf_aug_reest.y_filt[:, 2], label="Filtered")
ax[2].plot(idx, obs.iloc[:, 2], label="Actual", linestyle="--", alpha=0.7)
ax[2].set_title("Policy Rate")
ax[2].legend()

plt.tight_layout()
# %%
sim_aug_reest = sol_aug.sim(
    T=len(idx) - 1,
    shocks={"g,z": gz, "r": r},
    observables=True,
)
fig, ax = plt.subplots(3, 1, figsize=_FIGSIZE_1D)

plt.suptitle("Augmented + Re-estimated Model: Stochastic Simulation vs Actual")

ax[0].plot(idx, sim_aug_reest["OutGap"], label="Simulated")
ax[0].plot(idx, obs.iloc[:, 0], label="Actual", linestyle="--", alpha=0.7)
ax[0].set_title("OutGap")
ax[0].legend()

ax[1].plot(idx, sim_aug_reest["Infl"], label="Simulated")
ax[1].plot(idx, obs.iloc[:, 1], label="Actual", linestyle="--", alpha=0.7)
ax[1].set_title("Inflation")
ax[1].legend()

ax[2].plot(idx, sim_aug_reest["Rate"], label="Simulated")
ax[2].plot(idx, obs.iloc[:, 2], label="Actual", linestyle="--", alpha=0.7)
ax[2].set_title("Policy Rate")
ax[2].legend()

plt.tight_layout()
# %%
fig, ax = plt.subplots(3, 2, figsize=_FIGSIZE_2D)

plt.suptitle("Augmented + Re-estimated Model: Innovation AutoCorr")
plot_acf(
    kf_aug_reest.innov[:, 0],
    ax=ax[0, 0],
    lags=20,
    title="Output Gap Innovations ACF",
)
plot_acf(
    kf_aug_reest.innov[:, 1],
    ax=ax[1, 0],
    lags=20,
    title="Inflation Innovationsc ACF",
)
plot_acf(
    kf_aug_reest.innov[:, 2],
    ax=ax[2, 0],
    lags=20,
    title="Policy Rate Innovations ACF",
)
plot_pacf(
    kf_aug_reest.innov[:, 0],
    ax=ax[0, 1],
    lags=20,
    title="Output Gap Innovations PACF",
)
plot_pacf(
    kf_aug_reest.innov[:, 1],
    ax=ax[1, 1],
    lags=20,
    title="Inflation Innovations PACF",
)
plot_pacf(
    kf_aug_reest.innov[:, 2],
    ax=ax[2, 1],
    lags=20,
    title="Policy Rate Innovations PACF",
)

plt.tight_layout()
# %%
res_aug.hpd_intervals()
