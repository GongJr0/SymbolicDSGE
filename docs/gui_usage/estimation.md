---
tags:
    - guide
---

# Estimation Tab

The Estimation tab fits the active model's parameters to observed data using maximum likelihood, maximum a posteriori, or Markov chain Monte Carlo.

## Estimation Modes

Select one of three methods:

| Method | Description |
| --- | --- |
| MLE | Maximize the Kalman-filter log-likelihood. Priors are not required. |
| MAP | Maximize the log-posterior. Each estimated parameter requires a prior. |
| MCMC | Sample the posterior with a random-walk Metropolis sampler. Each estimated parameter requires a prior. |

MLE and MAP expose a maximum-iteration control for the optimizer. MCMC exposes the draw count, burn-in, thinning, seed, proposal scale, proposal adaptation, and the posterior point summary (mean, MAP, or last draw).

## Providing Data

Enter the observable column names, then supply one column of observations per name. Values accept newline-, comma-, space-, or semicolon-separated entries. **Import CSV** fills the columns from a CSV whose headers match the observable names.

## Configuring Parameters

The Parameters panel lists every model parameter. Select a parameter to edit it in the Estimation Details panel:

- **Estimate** toggles whether the parameter is included in the fit.
- **Initial** sets the starting value; **Lower** and **Upper** set optional bounds.
- **Prior** (MAP and MCMC only) selects a distribution, its parameters, and an optional transform.

At least one parameter must be selected, and the estimated parameter names must be unique.

## Running and Inspecting Results

Select **Run Estimation** to fit the model, or **Estimate & Solve** to also write the fitted values back to the active model and solve it.

The Latest Result section reports the method, success flag, and fitted values together with the log-likelihood, log-prior, and log-posterior. MCMC runs additionally display the acceptance rate, posterior summaries, the log-posterior trace, and per-parameter posterior distributions.

## Persistence and Clearing

The selected method, run settings, data columns, and parameter configuration persist across browser refreshes. **Clear** resets the workspace to its initial state.

???+ note "Local Persistence"
    Estimation workspace state is stored in the browser used to access the localhost GUI. It is not written into the model configuration.
