---
tags:
    - info
hide:
    - footer
---

# Introduction

`SymbolicDSGE` is a linear DSGE engine with a completely symbolic model specification. Through `SymPy`, model components are parsed into expressions that can be adjusted, decomposed, and analyzed. Flexible configurations are backed by native kernels that are optimized for small to medium scale models; SymbolicDSGE does not suffer from rnutime concerns associated with python and symbolic computation. The library API is currently unstable, with the stable `v2.0.0` release set to mark the first API freeze. Current features include:

- YAML-based model configuration
- Parser with a `SymPy` backend
- First and second order models with full support
- Solution and simulation for OBC models
- IRF path/plot generation
- Simulation
- Shock generation with built-in distributions and support for any custom distribution via tabulation.
- Bayesian prior API (`Prior`, `make_prior`) with transforms + distribution dispatch
- Estimation workflows (`MLE`, `MAP`, adaptive MH `MCMC`)
- Data retrieval helper for FRED API
- Data transformation functions (HP filters, detrending, etc.)
- Standalone fitlering API for multiple variants.
- Monte Carlo pipeline framework expressing multi-step experiments with complete native and parallel execution.
- OLS, Ridge, Lasso, and Elastic Net regression with coordinate descent solvers and grid search utilities with Monte Carlo regression steps
- Statistical test utilities with Monte Carlo test steps
