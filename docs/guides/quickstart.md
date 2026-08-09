---
tags:
    - guide
---

# Quick Start Guide

??? tip "__TL;DR__"
    You can find a demonstration notebook [here](../assets/guide_notebook.ipynb).

This guide will follow the steps necessary to get from model parsing to simulation.
We will use a pre-defined config file (accessible in the [repository](https://github.com/GongJr0/SymbolicDSGE/)) `"MODELS/POST82.yaml"`.


## Reading Model Configuration

The configuration files are parsed by the `#!python SymbolicDSGE.ModelParser` class.
The class provides `#!python .get()` (model only) and `#!python .get_all()` (model + kalman config).

```python
from SymbolicDSGE import ModelParser
from sympy import Matrix
from warnings import simplefilter, catch_warnings

parsed = ModelParser("MODELS/POST82.yaml").get_all()
model, kalman = parsed

with catch_warnings(): # (1)!
    simplefilter(action="ignore")
    mat = Matrix(model.equations.model)
mat
```

1. Wrapping equations in a `#!python sp.Matrix` is deprecated and used here solely for pretty-printing.

We've read the config and displayed the equations in a matrix:

$$
    \left[\begin{matrix}\Pi{\left(t \right)} = \beta \Pi{\left(t + 1 \right)} + \kappa x{\left(t \right)} + z{\left(t \right)}\\x{\left(t \right)} = - \tau_{inv} \left(- \Pi{\left(t + 1 \right)} + r{\left(t \right)}\right) + g{\left(t \right)} + x{\left(t + 1 \right)}\\r{\left(t \right)} = e_{R} + \rho_{r} r{\left(t - 1 \right)} + \left(1 - \rho_{r}\right) \left(\psi_{\pi} \Pi{\left(t \right)} + \psi_{x} x{\left(t \right)}\right)\\g{\left(t \right)} = e_{g} + \rho_{g} g{\left(t - 1 \right)}\\z{\left(t \right)} = e_{z} + \rho_{z} z{\left(t - 1 \right)}\end{matrix}\right]
$$

We can see that all variables are converted to `#!python SymPy` objects (symbols/functions) and are accessible through the `ModelConfig` interface.

## Compilation

In compilation, the symbolic model is projected into a functionalized and completely numeric form. Time dependent variables are separated and equations are written as lambda objectives. The first order solver consumes the compiled residual through the in house Klein pipeline.

If your model is written in nonlinear levels, pass `#!python linearize=True` to `#!python DSGESolver.compile(...)`. If you need the transformed symbolic equations directly, you can also call `#!python SymbolicDSGE.core.linearize_model(...)` yourself before compilation. The example below uses a model that is already written in linearized gap form.

```python
from SymbolicDSGE import DSGESolver

solver = DSGESolver(model, kalman)
compiled = solver.compile(
    variable_order = None, # (1)!
    params_order=None, # (2)!
    linearize=False, # (3)!
)

print("Equations with symbols removed: \n", "\n".join(map(str, compiled.objective_eqs)))
```

1. `#!python None | list[sp.Function | str]`. `#!python None` uses the variable order in the config file. Custom orders must declare `[*exog, *state, *control]` in that order. If groups are not contiguous or a different order is used, the compiler will raise a validation error. Within groups, any order is accepted.
2. `#!python None | list[str]`. `#!python None` uses the parameter order in the config file.
3. Set to `#!python True` to symbolically linearize the model config before compiling it.

At compilation, the equations are transformed as shown in the code output:

```text
Equations with symbols removed: 
 -beta*fwd_Pi + cur_Pi - kappa*(cur_x - cur_z)
-cur_g + cur_x - fwd_x + tau_inv*(cur_r - fwd_Pi)
-cur_r*rho_r + fwd_r + (rho_r - 1)*(fwd_Pi*psi_pi + psi_x*(fwd_x - fwd_z))
-cur_g*rho_g + fwd_g
-cur_z*rho_z + fwd_z
```

???+ note "Variable Layout"
    The compiler infers the canonical solver layout from the configuration. Shock-map targets form the shocked/exogenous state block, dynamic equations determine the remaining state variables, and the rest are controls.

???+ note "Linearization"
    When passing the linearization flag, the parsed `ModelConfig` must have the linearization parameters defined. (refer to the [Config Guide](./model_config_guide.md))
    Alternatively, you can import `SymbolicDSGE.linearize_model` to use the syntax `lin_config = linearize_model(my_config)`. 

## Solution

The solution step takes steady-state values and optionally parameter calibrations to provide a `#!python SolvedModel`.

```python
from numpy import float64, array

sol = solver.solve(
    compiled,
    parameters=None, # (1)!
    ss_seed=[0.0, 0.0, 0.0, 0.0, 0.0],
)
print("Is stable: ", sol.policy.stab == 0)  # (2)!
print("Eigenvalues: ", sol.policy.eig)
```

1. `#!python None | dict[str, float]`. `#!python None` uses the values in `#!python ModelConfig.calibration`
2. stable if `#!python sol.policy.stab == 0`

<div class="annotate" markdown>
```
Is stable:  True
Eigenvalues:  [0.28018451+0.j 0.83      +0.j 0.85      +0.j 2.60451546+0.j
 1.18546572+0.j] (1)
```
</div>
1. Eigenvalues stay complex because the ordered Schur solve makes the imaginary part meaningful here. The policy matrices are not: `#!python sol.policy.p` and `#!python sol.policy.f` are real, projected once inside the solve.

## Inspecting Model Dynamics

While we can check the matrices directly, we can also use the built-in methods `#!python SolvedModel.irf` and `#!python SolvedModel.transition_plot` to display the dynamics.

```python
irf_dict = sol.irf(
    T=25,
    shocks=["g", "z"],
    scale=1.0,  # (1)!
    observables=True,  # (2)!
)
sol.transition_plot(
    T=25,
    shocks=["g", "z"],
    scale=1.0,
    observables=True,
)
irf_dict["z"].round(3) # (3)!
```

1. `#!python shock = sig_var * scale`
2. Include observables in output.
3. Path of the variable `#!python z`.

This produces the outputs:
![transition_plot output](../img/qs_transition.png "transition_plot output")

```text
array([0.64 , 0.544, 0.462, 0.393, 0.334, 0.284, 0.241, 0.205, 0.174,
       0.148, 0.126, 0.107, 0.091, 0.077, 0.066, 0.056, 0.048, 0.04 ,
       0.034, 0.029, 0.025, 0.021, 0.018, 0.015, 0.013])
```

## Simulation

`#!python SolvedModel` also supplies a `#!python .sim()` method for simulations.
The method simulates `T` steps given an initial state array and a shock specification.

Shock specifications can take three basic forms.

- A `#!python Shock` distribution spec
- A callable returning the complete shock array: `#!python Callable[[float | ndarray], ndarray]`
- A `#!python np.ndarray` of innovations

Any specification is delivered to `.sim` in a dictionary corresponding to the variable the innovations are meant to affect.
In case of multiple shocks with correlation the key for the dictionary uses `"g,z"` syntax. In correlated cases, `Shock` and callable values receive the model covariance matrix, while array values must have shape `(T, n_correlated_shocks)`.

`SymbolicDSGE.Shock` is an interface simplifying the shock generation process. It can be passed directly to `.sim`, which materializes a `T` period draw at simulation time. The class has support for all `SciPy` distributions from the `rv_generic` and `multi_rv_generic` hierarchies. Alongside `SciPy` support, custom distributions implementing the `.rvs` method are supported through distribution `args`/`kwargs`.

```python
from SymbolicDSGE import Shock

T = 200
multi_shock_spec = lambda seed: Shock(
    dist="norm",
    multivar=True,
    seed=seed,
    dist_kwargs={
        "mean": [0.0, 0.0],
    },
)

uni_shock_spec = lambda seed: Shock(
    dist="norm",
    multivar=False,
    seed=seed,
    dist_kwargs={
        "loc": 0.0,
    },
)

sim_shocks = {
    "g,z": multi_shock_spec(seed=1),
    "r": uni_shock_spec(seed=2),
}

```

1. Notice the seed argument to the class being parametrized through a lambda. This step is not necessary for functionality. It saves the code of declaring two instances with different seeds if two shocks share distributions.
2. Seed is passed through here, the code below would operate the same if we used `seed=1` instead of using a lambda.
3. The `kwargs` specified here are passed to the distribution object in the backend (to `SciPy`'s `rvs` methods in this case)
4. The value in this pair is a `Shock` object. `.sim` supplies the horizon and constructs the appropriate standard deviation or covariance from model parameters.

With the shocks specified, we can simulate stochastic paths as follows:

```python
import pandas as pd

sim_data = sol.sim(
    T=T,
    x0=[0.0, 0.0, 0.0, 0.0, 0.0],  # (1)!
    shocks=sim_shocks,
    shock_scale=1.0,
    observables=True,
)
del sim_data["_X"]  # (2)!
pd.DataFrame(sim_data).head(10).round(3)  
```

1. Simulation starts at steady state
2. `"_X"` is a `ndarray` of all non-observable states for each time t. It is deleted here for code brevity in producing a `DataFrame`.

|    |      g |      z |      r |      x |     Pi |   OutGap |   Infl |   Rate |
|---:|-------:|-------:|-------:|-------:|-------:|---------:|-------:|-------:|
|  0 |  0.062 |  0.57  |  0.034 |  0.272 | -0.239 |    0.272 |  2.472 |  6.576 |
|  1 |  0.111 | -0.217 | -0.094 |  0.819 |  0.825 |    0.819 |  6.729 |  6.066 |
|  2 |  0.255 |  0.29  | -0.058 |  1.314 |  0.812 |    1.314 |  6.678 |  6.207 |
|  3 |  0.115 |  0.47  | -0.396 |  3.018 |  2.028 |    3.018 | 11.54  |  4.856 |
|  4 |  0.161 |  0.659 |  0.224 | -0.528 | -0.949 |   -0.528 | -0.366 |  7.336 |
|  5 |  0.139 |  0.893 |  0.284 | -0.85  | -1.393 |   -0.85  | -2.141 |  7.576 |
|  6 | -0.017 |  0.492 |  0.019 |  0.073 | -0.335 |    0.073 |  2.089 |  6.515 |
|  7 | -0.101 |  0.665 |  0.116 | -0.705 | -1.092 |   -0.705 | -0.938 |  6.905 |
|  8 | -0.077 |  0.4   |  0.023 | -0.187 | -0.467 |   -0.187 |  1.562 |  6.531 |
|  9 | -0.204 |  0.006 | -0.134 |  0.17  |  0.133 |    0.17  |  3.964 |  5.903 |

Alternative to a DataFrame, we can also plot the simulated paths:

```python
from numpy import ceil, sqrt
import matplotlib.pyplot as plt

fig_square = ceil(sqrt(len(sim_data))).astype(int)
size = (4 * fig_square, 3 * fig_square)
fig, ax = plt.subplots(fig_square, fig_square, figsize=size)
ax = ax.flatten()

while len(ax) > len(sim_data):
    fig.delaxes(ax[-1])
    ax = ax[:-1]

for i, (var, path) in enumerate(sim_data.items()):
    ax[i].plot(path)
    ax[i].set_title(var)
    ax[i].grid(linestyle=":")
plt.suptitle(f"Simulation over {T} periods with stochastic shocks", fontsize=16)
plt.tight_layout()
```

![Simulation Plots](../img/qs_sim.png "Simulated Paths")

## Further Steps

This guide covers the basic capabilities and usage of `SymbolicDSGE`. Further tools include:

- `SymbolicDSGE.utils.FRED` for easy U.S. macro data retrieval
- `SymbolicDSGE.utils.math_utils` for basic detrending, HP filters, etc.
- `SymbolicDSGE.kalman` (integrated with `SolvedModel` via `SolvedModel.kalman`) for state estimation and filtering through KF, EKF, and UKF methods.

If you've read to this point and would like to inspect/interact with the code this guide refers to, you can visit [this](../assets/guide_notebook.ipynb) link to the file.

[Download Guide Notebook](../assets/guide_notebook.ipynb){ .md-button download="" }
