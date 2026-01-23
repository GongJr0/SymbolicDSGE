---
tags:
    - info
hide:
    - footer
---

# Introduction

`SymbolicDSGE` is a linear DSGE engine with a completely symbolic model specification. Through `SymPy`, model components are parsed into expressions that can be adjusted, decomposed, and analyzed. This allows things like searching model parameters in a grid, quickly modifying and testing parsed equations, and more; All parsed components of the model support overriding and recompiling. Although the library is currently in very early development, current functionality includes:

- YAML-based model configuration
- Parser with a `SymPy` backend
- `linearsolve` based solver
- IRF path/plot generation
- Simulation
- Shock generation interface with support for all `SciPy` distributions
- Data retrieval helper for FRED API
- Data transformation functions (HP filters, detrending, etc.)
- Kalman Filter implementation