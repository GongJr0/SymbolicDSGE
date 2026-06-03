---
tags:
    - info
hide:
    - footer
---

# Introduction

`SymbolicDSGE` is a linear DSGE engine with a completely symbolic model specification. Through `SymPy`, model components are parsed into expressions that can be adjusted, decomposed, and analyzed. This allows things like searching model parameters in a grid, quickly modifying and testing parsed equations, and more; All parsed components of the model support overriding and recompiling. The library is actively being developed, and features are expanding quickly. Current functionality includes:

- YAML-based model configuration
- Parser with a `SymPy` backend
- `linearsolve` based solver
- IRF path/plot generation
- Simulation
- Shock generation interface with support for all `SciPy` distributions
- Bayesian prior API (`Prior`, `make_prior`) with transforms + distribution dispatch
- Estimation workflows (`MLE`, `MAP`, adaptive Metropolis `MCMC`)
- Data retrieval helper for FRED API
- Data transformation functions (HP filters, detrending, etc.)
- Kalman Filter implementation
- Monte Carlo pipeline framework with support for user-defined steps, context objects, and result containers
- OLS, Ridge, Lasso, and Elastic Net regression with coordinate descent solvers and grid search utilities with Monte Carlo regression steps
- Statistical test utilities with Monte Carlo test steps
