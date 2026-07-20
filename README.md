<a href="https://gongjr0.github.io/SymbolicDSGE/">
  <picture>
    <!-- Dark mode -->
    <source srcset="docs/assets/logo_w.svg" media="(prefers-color-scheme: dark)">
    <!-- Light mode -->
    <source srcset="docs/assets/logo_b.svg" media="(prefers-color-scheme: light)">
    <!-- Fallback -->
    <img src="docs/assets/logo_b.svg" alt="SymbolicDSGE logo">
  </picture>
</a>
<h4 align="right">by Güney Kıymaç</h4>
<hr>

<div align="center">
  <a href="https://github.com/GongJr0/SymbolicDSGE/actions"><img src="https://raw.githubusercontent.com/GongJr0/SymbolicDSGE/badges/tests-badge.svg" alt="tests"></a>
  &nbsp;
  <a href="https://github.com/GongJr0/SymbolicDSGE/actions"><img src="https://raw.githubusercontent.com/GongJr0/SymbolicDSGE/badges/pre-commit-badge.svg" alt="pre-commit"></a>
  &nbsp;
  <a href="https://github.com/GongJr0/SymbolicDSGE/actions"><img src="https://raw.githubusercontent.com/GongJr0/SymbolicDSGE/badges/coverage-badge.svg" alt="coverage"></a>
  <a href="https://pepy.tech/projects/symbolicdsge"><img src="https://static.pepy.tech/personalized-badge/symbolicdsge?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=ORANGE&left_text=downloads" alt="PyPI Downloads"></a>
</div>
<div align="center">
  <a href="https://doi.org/10.5281/zenodo.20115401"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20115401-blue" alt="DOI"></a>
  &nbsp;
  <a href="https://wakatime.com/badge/github/GongJr0/SymbolicDSGE"><img src="https://wakatime.com/badge/github/GongJr0/SymbolicDSGE.svg" alt="wakatime"></a>
</div>

<div align="right">
  <a href="https://gongjr0.github.io/SymbolicDSGE/">Documentation</a>
  &nbsp; | &nbsp;
  <a href="https://pypi.org/project/SymbolicDSGE/">PyPI</a>
</div>
<br>

### Installation
```bash
pip install SymbolicDSGE
pip install "SymbolicDSGE[fred]"  # FRED API utilities
pip install "SymbolicDSGE[sr]"    # Symbolic Regression Deps
pip install "SymbolicDSGE[ui]"    # Web based GUI
```

### Useful Links
- [Installation](https://gongjr0.github.io/SymbolicDSGE/latest/installation)
- [Quick Start Guide](https://gongjr0.github.io/SymbolicDSGE/latest/Guides/quickstart/)

### Overview

`SymbolicDSGE` is a Python DSGE engine with a JIT compiled backend for linear and linearized DSGE models, supporting symbolic manipulation features for in-place model modification.
It also provides measurement-equation augmentation tools, including symbolic regression for complete or restricted free-form function discovery and OLS, Ridge, Lasso, and Elastic Net for structured linear coefficient estimation.
The library supports a wide set of features beyond augmentation:

- DSGE model specification, symbolic manipulation, and linearization
- Bayesian and maximum-likelihood estimation
- Simulation and impulse-response-function utilities
- Gaussian and extended Kalman filtering
- Automatic data retrieval from FRED
- Custom shock distributions via SciPy or user-defined samplers
- Monte Carlo pipelines for comparing models or model-generated samples, including:
  - Statistical tests
  - Regression setups
  - Output transformations
  - Filtering
  - Data generation

### Web-UI

`SymbolicDSGE` ships a `[ui]` dependency with the goal of making DSGE basic experiments accessible to non-programmers.
GUI users can:

- Create a model config
- Load and adjust existing configs
- Estimate
- Solve
- Simulate
- Create and run Monte Carlo experiments
- Apply data transformations to outputs
- Produce figures
- Access raw output data

There are two intended ways to access the GUI:

```bash
# Option 1: Serve the GUI from the command line
#           This launches an empty GUI and the configuration should be done through the Builder tab.
sdsge-ui
```

```python
# Option 2: Serve the GUI with a pre-loaded SolvedModel object

# Read a config and solve the model as usual
from SymbolicDSGE import ModelParser, DSGESolver

model, kalman = ModelParser(...).get_all()
solver = DSGESolver(model, kalman)
compiled = solver.compile(...)
sol = solver.solve(compiled, ...)

# Serve the GUI with the solved model pre-loaded in the reference slot
sol.serve(open_browser=True)
```

Although SymbolicDSGE is still a fast-moving project; the UI effectively supports all features with the exception of integration with the `[sr]` extra as of `v1.4.2`. Symbolic Regression requires extensive configuration and templating posing unique difficulties in a no-code environment. Therefore, it's presence as a GUI component is not planned for the future.

### Read the Docs

Alongside API references and implementation conventions, the [documentation](https://gongjr0.github.io/SymbolicDSGE/latest/) includes guides covering model setup, estimation, simulation, and filtering.
The documentation is kept up to date and aims to clarify conventions, workflows, and implementation choices throughout `SymbolicDSGE`.
Suggestions for improving or extending the documentation are welcome as issues.

> __AI Parsability__
>
> The documentation is a static Material MkDocs site, so web-enabled AI tools should be able to search, parse, and summarize it reliably.

### Minimal Example

```python
from SymbolicDSGE import ModelParser
from SymbolicDSGE import DSGESolver

# Read the YAML config (Equations, Measurements, Parameters, Optional Filter Spec)
parsed = ModelParser("<path-to-config>.yaml").get_all()
model, kalman = parsed

# Compile the model
solver = DSGESolver(model, kalman)
compiled = solver.compile()

print("Equations with symbols removed: \n", "\n".join(map(str, compiled.objective_eqs)), "\n")
```
```text
>>> Equations with symbols removed:
 -beta*fwd_Pi + cur_Pi - cur_x*kappa - cur_z
-cur_g + cur_x - fwd_x + tau_inv*(cur_r - fwd_Pi)
-cur_r*rho_r - e_R + fwd_r + (rho_r - 1)*(fwd_Pi*psi_pi + fwd_x*psi_x)
-cur_g*rho_g - e_g + fwd_g
-cur_z*rho_z - e_z + fwd_z
```
```python
# Solve the compiled model
sol = solver.solve(
    compiled,
    parameters=None,
    steady_state=[0.0, 0.0, 0.0, 0.0, 0.0],
)
print("Is stable: ", sol.policy.stab == 0)
print("Eigenvalues: ", sol.policy.eig)
```
```text
>>> Is stable:  True
Eigenvalues:  [0.27920118+0.j 0.83000003+0.j 0.84999992+0.j 2.56517116+0.j 1.18470582+0.j]
```
```python
# Plot IRFs (single or multi shock)
sol.transition_plot(
    T=25,
    shocks=["g", "z"],
    scale=1.0,
    observables=True,
)
```
<img width="3570" height="2661" alt="image" src="https://github.com/user-attachments/assets/f5931ad7-a70b-43e9-b42e-614bc32f3854" />
