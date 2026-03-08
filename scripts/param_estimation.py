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
from SymbolicDSGE.utils.math_utils import HP_two_sided, annualized_log_percent, detrend
from SymbolicDSGE.bayesian import make_prior
from SymbolicDSGE.regression import (
    SymbolicRegressor,
    TemplateConfig,
    PySRParams,
    ModelParametrizer,
)

from sympy import Matrix, Symbol, print_latex, Rational
from warnings import catch_warnings, simplefilter

from numpy import array, float64, ceil, sqrt, log, std, random, isclose
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# %%
model, kalman = ModelParser("MODELS/POST82.yaml").get_all()

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
        "GDPC1",  # Real GDP
        "CPIAUCSL",  # Consumer Price Index for All Urban Consumers: All Items
        "FEDFUNDS",  # Effective Federal Funds Rate
    ],
    date_range=(
        "1955-01-01",
        "2007-10-01",
    ),  # Date range for the data ("YYYY-MM-DD" format or a pd.DatetimeIndex object)
)

gdp_q = df["GDPC1"]  # already quarterly in most pulls; verify freq

cpi_q = df["CPIAUCSL"].resample("QS").mean()  # quarterly avg CPI
ffr_q = df["FEDFUNDS"].resample("QS").mean()  # quarterly avg policy rate

idx_range = pd.date_range(start="1984-01-01", end="2007-01-01", freq="QS")


df = pd.DataFrame(
    {
        "GDPC1": gdp_q.reindex(idx_range),
        "CPIAUCSL": cpi_q.reindex(idx_range),
        "FEDFUNDS": ffr_q.reindex(idx_range),
    }
)

df

# %%
x_trend = HP_two_sided(log(df["GDPC1"]), lamb=1600)[0]  # returns (trend, cycle)
x = (log(df["GDPC1"]) - x_trend) * 100  # HP detrended quarterly log output gap


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
        "Infl": inf_lvl[df_model_units.index],
        "Rate": rate_lvl[df_model_units.index],
    }
)
observed.index = df_model_units.index

# %%
prior_spec = {
    # (0, 1)
    "beta": make_prior(
        "beta",
        parameters={"a": 200 * 0.99, "b": 200 * 0.001},
        transform="logit",
    ),
    "rho_r": make_prior(
        "beta",
        parameters={"a": 200 * 0.84, "b": 200 * 0.16},
        transform="logit",
    ),
    "rho_g": make_prior(
        "beta",
        parameters={"a": 200 * 0.83, "b": 200 * 0.17},
        transform="logit",
    ),
    "rho_z": make_prior(
        "beta",
        parameters={"a": 200 * 0.85, "b": 200 * 0.15},
        transform="logit",
    ),
    # (0, +inf)
    "psi_pi": make_prior(
        "gamma",
        parameters={"mean": 2.19, "std": 0.5},
        transform="log",
    ),
    "psi_x": make_prior(
        "gamma",
        parameters={"mean": 0.30, "std": 0.1},
        transform="log",
    ),
    "kappa": make_prior(
        "gamma",
        parameters={"mean": 0.58, "std": 0.1},
        transform="log",
    ),
    "tau_inv": make_prior(
        "gamma",
        parameters={"mean": 1.86, "std": 0.5},
        transform="log",
    ),
    # Correlation (-1,1)
    "rho_gz": make_prior(
        "normal",
        parameters={"mean": 0.0, "std": 0.20},
        transform="affine_logit",
        transform_kwargs={
            "low": -1.0,
            "high": 1.0,
        },
    ),
    "meas_rho_ir": make_prior(
        "normal",
        parameters={"mean": 0.0, "std": 0.4},
        transform="affine_logit",
        transform_kwargs={"low": -1.0, "high": 1.0},
    ),
    # Shock std devs (0, +inf)
    "sig_r": make_prior(
        "gamma",
        parameters={"mean": 0.18, "std": 0.1},
        transform="log",
    ),
    "sig_g": make_prior(
        "gamma",
        parameters={"mean": 0.18, "std": 0.1},
        transform="log",
    ),
    "sig_z": make_prior(
        "gamma",
        parameters={"mean": 0.64, "std": 0.1},
        transform="log",
    ),
    "meas_infl": make_prior(
        "normal",
        parameters={"mean": 0.0, "std": 0.4},
        transform="log",
    ),
    "meas_rate": make_prior(
        "normal",
        parameters={"mean": 0.0, "std": 0.4},
        transform="log",
    ),
}

solver = DSGESolver(model, kalman)
comp = solver.compile(
    n_exog=3,
    n_state=3,
)
estim = lambda r: solver.estimate_and_solve(
    compiled=comp,
    y=observed.loc[observed.index >= "1984-01-01", :],
    priors=prior_spec,
    method="mcmc",
    posterior_point="mean",
    steady_state=[0.0, 0.0, 0.0, 0.0, 0.0],
    estimated_params=list(prior_spec.keys()),
    n_draws=1000,
    burn_in=500,
    thin=1,
    update_R_in_iterations=r,
)
res, sol = estim(False)

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
res1, sol1 = estim(True)

# %%
param_names1 = res1.param_names

post_mean1 = np.mean(res1.samples, axis=0)

loglik1 = np.mean(res1.logpost_trace)
accept_rate1 = res1.accept_rate
pd.Series(
    {
        **dict(zip(param_names1, post_mean1)),
        "loglik": loglik1,
        "accept_rate": accept_rate1,
        "n_draws": res1.n_draws,
        "burn_in": res1.burn_in,
        "thin": res1.thin,
    }
)

# %%
sol1.transition_plot(shocks=["g"], T=25, scale=1, observables=True)

# %%
kf_0 = sol.kalman(
    observed.loc[observed.index >= "1984-01-01", :],
    "linear",
    return_shocks=True,
)
kf_1 = sol1.kalman(
    observed.loc[observed.index >= "1984-01-01", :],
    "linear",
    return_shocks=True,
)

# %%
obs = observed.loc[observed.index >= "1984-01-01", :]
idx = obs.index

fig, ax = plt.subplots(2, 2, figsize=(12, 6))
ax = ax.flatten()

plt.suptitle("[Static R] Filtered, Predicted vs Actual Measurements")

ax[0].plot(idx, kf_0.y_pred[:, 0], label="Predicted")
ax[0].plot(idx, obs.iloc[:, 0], label="Actual", linestyle="--", alpha=0.7)
ax[0].set_title("Inflation")
ax[0].legend()

ax[1].plot(idx, kf_0.y_pred[:, 1], label="Predicted")
ax[1].plot(idx, obs.iloc[:, 1], label="Actual", linestyle="--", alpha=0.7)
ax[1].set_title("Policy Rate")
ax[1].legend()

ax[2].plot(idx, kf_0.y_filt[:, 0], label="Filtered")
ax[2].plot(idx, obs.iloc[:, 0], label="Actual", linestyle="--", alpha=0.7)
ax[2].set_title("Inflation (Filtered)")
ax[2].legend()

ax[3].plot(idx, kf_0.y_filt[:, 1], label="Filtered")
ax[3].plot(idx, obs.iloc[:, 1], label="Actual", linestyle="--", alpha=0.7)
ax[3].set_title("Policy Rate (Filtered)")
ax[3].legend()

plt.tight_layout()

# %%
fig, ax = plt.subplots(2, 2, figsize=(12, 6))
ax = ax.flatten()

plt.suptitle("[Dynamic R] Filtered, Predicted vs Actual Measurements")

ax[0].plot(idx, kf_1.y_pred[:, 0], label="Predicted")
ax[0].plot(idx, obs.iloc[:, 0], label="Actual", linestyle="--", alpha=0.7)
ax[0].set_title("Inflation")
ax[0].legend()

ax[1].plot(idx, kf_1.y_pred[:, 1], label="Predicted")
ax[1].plot(idx, obs.iloc[:, 1], label="Actual", linestyle="--", alpha=0.7)
ax[1].set_title("Policy Rate")
ax[1].legend()

ax[2].plot(idx, kf_1.y_filt[:, 0], label="Filtered")
ax[2].plot(idx, obs.iloc[:, 0], label="Actual", linestyle="--", alpha=0.7)
ax[2].set_title("Inflation (Filtered)")
ax[2].legend()

ax[3].plot(idx, kf_1.y_filt[:, 1], label="Filtered")
ax[3].plot(idx, obs.iloc[:, 1], label="Actual", linestyle="--", alpha=0.7)
ax[3].set_title("Policy Rate (Filtered)")
ax[3].legend()

plt.tight_layout()

# %%
gz = Shock(T=len(idx) - 1, dist="norm", multivar=True, seed=0).shock_generator()
r = Shock(T=len(idx) - 1, dist="norm", seed=1).shock_generator()
sim0 = sol.sim(
    T=len(idx) - 1,
    shocks={"g,z": gz, "r": r},
    observables=True,
)

sim1 = sol1.sim(
    T=len(idx) - 1,
    shocks={"g,z": gz, "r": r},
    observables=True,
)

# %%
fig, ax = plt.subplots(2, 2, figsize=(12, 6))
ax = ax.flatten()

plt.suptitle("Stochastic Simulation vs Actual")

ax[0].plot(idx, sim0["Infl"], label="Simulated (Static R)")
ax[0].plot(idx, obs.iloc[:, 0], label="Actual", linestyle="--", alpha=0.7)
ax[0].set_title("Inflation")
ax[0].legend()

ax[1].plot(idx, sim0["Rate"], label="Simulated (Static R)")
ax[1].plot(idx, obs.iloc[:, 1], label="Actual", linestyle="--", alpha=0.7)
ax[1].set_title("Policy Rate")
ax[1].legend()

ax[2].plot(idx, sim1["Infl"], label="Simulated (Dynamic R)")
ax[2].plot(idx, obs.iloc[:, 0], label="Actual", linestyle="--", alpha=0.7)
ax[2].set_title("Inflation")
ax[2].legend()

ax[3].plot(idx, sim1["Rate"], label="Simulated (Dynamic R)")
ax[3].plot(idx, obs.iloc[:, 1], label="Actual", linestyle="--", alpha=0.7)
ax[3].set_title("Policy Rate")
ax[3].legend()

plt.tight_layout()

# %%
fig, ax = plt.subplots(2, 2, figsize=(12, 6))
ax = ax.flatten()

plt.suptitle("Innovation AutoCorr")
plot_acf(kf_0.innov[:, 0], ax=ax[0], lags=20, title="Inflation Innovations (Static R)")
plot_acf(
    kf_0.innov[:, 1], ax=ax[1], lags=20, title="Policy Rate Innovations (Static R)"
)
plot_acf(kf_1.innov[:, 0], ax=ax[2], lags=20, title="Inflation Innovations (Dynamic R)")
plot_acf(
    kf_1.innov[:, 1], ax=ax[3], lags=20, title="Policy Rate Innovations (Dynamic R)"
)
plt.tight_layout()

# %%
fig, ax = plt.subplots(2, 2, figsize=(12, 6))
ax = ax.flatten()

plt.suptitle("Innovation Partial AutoCorr")
plot_pacf(kf_0.innov[:, 0], ax=ax[0], lags=20, title="Inflation Innovations (Static R)")
plot_pacf(
    kf_0.innov[:, 1], ax=ax[1], lags=20, title="Policy Rate Innovations (Static R)"
)
plot_pacf(
    kf_1.innov[:, 0], ax=ax[2], lags=20, title="Inflation Innovations (Dynamic R)"
)
plot_pacf(
    kf_1.innov[:, 1], ax=ax[3], lags=20, title="Policy Rate Innovations (Dynamic R)"
)
plt.tight_layout()

# %%
template = TemplateConfig(
    include_expression=False,
    interaction_form="func",
    hessian_restriction="free",
    power_law_lower_bound=2,
    power_law_upper_bound=2,
    powers_in_interactions=False,
)

params = PySRParams(
    niterations=200,
    maxsize=12,
    complexity_of_constants=3,
    complexity_of_variables=1,
    deterministic=True,
    random_state=0,
    parallelism="serial",
)

res = sol1.fit_kf(
    template_config=template,
    sr_params=params,
    y=observed.loc[observed.index >= "1984-01-01", :],
    variables=["r", "Pi", "x"],
    observable="Rate",
).expressions


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


sum_moments(pd.Series(kf_0.y_filt[:, 1]))

# %%
print(res[["sympy_format", "loss", "complexity"]].to_latex())

# %%

# Augmented Model
conf_aug, kalman_aug = ModelParser("./papers/progress/report1/Augmented.yaml").get_all()
solver_aug = DSGESolver(conf_aug, kalman_aug)
comp_aug = solver_aug.compile(n_exog=3, n_state=3)

aug = solver_aug.solve(compiled=comp_aug)  # Solve without re-estimation

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

fig, ax = plt.subplots(2, 2, figsize=(12, 6))
ax = ax.flatten()

plt.suptitle("Augmented Model: Filtered, Predicted vs Actual Measurements")

ax[0].plot(idx, kf_aug.y_pred[:, 0], label="Predicted")
ax[0].plot(idx, obs.iloc[:, 0], label="Actual", linestyle="--", alpha=0.7)
ax[0].set_title("Inflation")
ax[0].legend()

ax[1].plot(idx, kf_aug.y_pred[:, 1], label="Predicted")
ax[1].plot(idx, obs.iloc[:, 1], label="Actual", linestyle="--", alpha=0.7)
ax[1].set_title("Policy Rate")
ax[1].legend()

ax[2].plot(idx, kf_aug.y_filt[:, 0], label="Filtered")
ax[2].plot(idx, obs.iloc[:, 0], label="Actual", linestyle="--", alpha=0.7)
ax[2].set_title("Inflation (Filtered)")
ax[2].legend()

ax[3].plot(idx, kf_aug.y_filt[:, 1], label="Filtered")
ax[3].plot(idx, obs.iloc[:, 1], label="Actual", linestyle="--", alpha=0.7)
ax[3].set_title("Policy Rate (Filtered)")
ax[3].legend()

plt.tight_layout()
# %%
sim_aug = aug.sim(
    T=len(idx) - 1,
    shocks={"g,z": gz, "r": r},
    observables=True,
)
fig, ax = plt.subplots(2, 1, figsize=(12, 6))

plt.suptitle("Augmented Model: Stochastic Simulation vs Actual")

ax[0].plot(idx, sim_aug["Infl"], label="Simulated")
ax[0].plot(idx, obs.iloc[:, 0], label="Actual", linestyle="--", alpha=0.7)
ax[0].set_title("Inflation")
ax[0].legend()

ax[1].plot(idx, sim_aug["Rate"], label="Simulated")
ax[1].plot(idx, obs.iloc[:, 1], label="Actual", linestyle="--", alpha=0.7)
ax[1].set_title("Policy Rate")
ax[1].legend()

plt.tight_layout()

# %%
fig, ax = plt.subplots(2, 2, figsize=(12, 6))
ax = ax.flatten()

plt.suptitle("Augmented Model: Innovation AutoCorr")
plot_acf(kf_aug.innov[:, 0], ax=ax[0], lags=20, title="Inflation Innovations ACF")
plot_acf(kf_aug.innov[:, 1], ax=ax[1], lags=20, title="Policy Rate Innovationsc ACF")

plot_pacf(kf_aug.innov[:, 0], ax=ax[2], lags=20, title="Inflation Innovations PACF")
plot_pacf(kf_aug.innov[:, 1], ax=ax[3], lags=20, title="Policy Rate Innovations PACF")
plt.tight_layout()
# %%

sum_moments(pd.Series(sim_aug["Infl"]))

# %%
# Augmented + Re-estimated model

res_aug, sol_aug = solver_aug.estimate_and_solve(
    compiled=comp_aug,
    y=observed.loc[observed.index >= "1984-01-01", :],
    priors=prior_spec,
    method="mcmc",
    posterior_point="mean",
    steady_state=[0.0, 0.0, 0.0, 0.0, 0.0],
    estimated_params=list(prior_spec.keys()),
    n_draws=1000,
    burn_in=500,
    thin=1,
)
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

fig, ax = plt.subplots(2, 2, figsize=(12, 6))
ax = ax.flatten()

plt.suptitle(
    "Augmented + Re-estimated Model: Filtered, Predicted vs Actual Measurements"
)

ax[0].plot(idx, kf_aug_reest.y_pred[:, 0], label="Predicted")
ax[0].plot(idx, obs.iloc[:, 0], label="Actual", linestyle="--", alpha=0.7)
ax[0].set_title("Inflation")
ax[0].legend()

ax[1].plot(idx, kf_aug_reest.y_pred[:, 1], label="Predicted")
ax[1].plot(idx, obs.iloc[:, 1], label="Actual", linestyle="--", alpha=0.7)
ax[1].set_title("Policy Rate")
ax[1].legend()

ax[2].plot(idx, kf_aug_reest.y_filt[:, 0], label="Filtered")
ax[2].plot(idx, obs.iloc[:, 0], label="Actual", linestyle="--", alpha=0.7)
ax[2].set_title("Inflation (Filtered)")
ax[2].legend()

ax[3].plot(idx, kf_aug_reest.y_filt[:, 1], label="Filtered")
ax[3].plot(idx, obs.iloc[:, 1], label="Actual", linestyle="--", alpha=0.7)
ax[3].set_title("Policy Rate (Filtered)")
ax[3].legend()

plt.tight_layout()
# %%
sim_aug_reest = sol_aug.sim(
    T=len(idx) - 1,
    shocks={"g,z": gz, "r": r},
    observables=True,
)
fig, ax = plt.subplots(2, 1, figsize=(12, 6))

plt.suptitle("Augmented + Re-estimated Model: Stochastic Simulation vs Actual")

ax[0].plot(idx, sim_aug_reest["Infl"], label="Simulated")
ax[0].plot(idx, obs.iloc[:, 0], label="Actual", linestyle="--", alpha=0.7)
ax[0].set_title("Inflation")
ax[0].legend()

ax[1].plot(idx, sim_aug_reest["Rate"], label="Simulated")
ax[1].plot(idx, obs.iloc[:, 1], label="Actual", linestyle="--", alpha=0.7)
ax[1].set_title("Policy Rate")
ax[1].legend()

plt.tight_layout()
# %%
fig, ax = plt.subplots(2, 2, figsize=(12, 6))
ax = ax.flatten()

plt.suptitle("Augmented + Re-estimated Model: Innovation AutoCorr")
plot_acf(kf_aug_reest.innov[:, 0], ax=ax[0], lags=20, title="Inflation Innovations ACF")
plot_acf(
    kf_aug_reest.innov[:, 1], ax=ax[1], lags=20, title="Policy Rate Innovationsc ACF"
)
plot_pacf(
    kf_aug_reest.innov[:, 0], ax=ax[2], lags=20, title="Inflation Innovations PACF"
)
plot_pacf(
    kf_aug_reest.innov[:, 1], ax=ax[3], lags=20, title="Policy Rate Innovations PACF"
)
plt.tight_layout()
# %%
res_aug.hpd_intervals()
