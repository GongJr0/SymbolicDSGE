---
tags:
    - doc
---
# Prior

```python
from SymbolicDSGE.bayesian import Prior, make_prior
```

`Prior` combines a distribution and a transform into a prior defined on unconstrained space.

__Core API:__

| __Name__ | __Signature__ | __Description__ |
|:---------|:--------------|----------------:|
| `logpdf` | `#!python Prior.logpdf(z)` | Log-density in transformed space. |
| `grad_logpdf` | `#!python Prior.grad_logpdf(z)` | Gradient of log-density in transformed space. |
| `rvs` | `#!python Prior.rvs(size, random_state)` | Sampling in distribution space. |
| `support` | `#!python Prior.support` | Distribution support. |
| `maps_to` | `#!python Prior.maps_to` | Transform target support. |

&nbsp;

```python
make_prior(
    distribution: str, # (1)!
    parameters: dict[str, Any], # (2)!
    transform: str, # (3)!
    transform_kwargs: dict[str, Any] | None = None, # (4)!
) -> Prior
```

1. Distribution family string from `DistributionFamily`.
2. Parameter overrides merged with distribution defaults.
3. Transform method string from `TransformMethod` dispatch.
4. Extra kwargs for transform constructors that require bounds.

???+ note "Space Convention"
    `Prior.logpdf(z)` expects `z` in transformed (typically unconstrained) space. Internally it inverse-transforms to parameter space and applies Jacobian correction.

## Distribution Families
Accepted `distribution` values in `make_prior(...)`:

| __Enum Member__ | __String__ | __Parameter Keys__ | __Defaults__ |
|:----------------|:----------:|-------------------:|-------------:|
| `NORMAL` | `"normal"` | `mean`, `std`, `random_state` | `0.0`, `1.0`, `None` |
| `LOGNORMAL` | `"log_normal"` | `s`, `low`, `scale`, `random_state` | `1.0`, `0.0`, `1.0`, `None` |
| `HALFNORMAL` | `"half_normal"` | `low`, `scale`, `random_state` | `0.0`, `1.0`, `None` |
| `TRUNCNORMAL` | `"trunc_normal"` | `low`, `high`, `loc`, `scale`, `random_state` | `-6.0`, `6.0`, `0.0`, `1.0`, `None` |
| `HALFCAUCHY` | `"half_cauchy"` | `low`, `scale`, `random_state` | `0.0`, `1.0`, `None` |
| `BETA` | `"beta"` | `a`, `b`, `loc`, `scale`, `random_state` | `1.0`, `1.0`, `0.0`, `1.0`, `None` |
| `GAMMA` | `"gamma"` | `a`, `loc`, `scale`, `random_state` | `1.0`, `0.0`, `1.0`, `None` |
| `INVGAMMA` | `"inv_gamma"` | `a`, `loc`, `scale`, `random_state` | `1.0`, `0.0`, `1.0`, `None` |
| `UNIFORM` | `"uniform"` | `low`, `high`, `random_state` | `0.0`, `1.0`, `None` |
| `LKJCHOL` | `"lkj_chol"` | `eta`, `K`, `random_state` | `1.0`, `-1`, `None` |

???+ note "LKJ Parameter"
    `LKJCHOL` requires `K` to be set meaningfully by the user; the default `K=-1` is only a placeholder and is not a valid runtime shape.

## Transform Methods
Dispatched `transform` values accepted by `make_prior(...)`:

| __Enum Member__ | __String__ | __transform_kwargs__ |
|:----------------|:----------:|---------------------:|
| `IDENTITY` | `"identity"` | none |
| `LOG` | `"log"` | none |
| `SOFTPLUS` | `"softplus"` | none |
| `LOGIT` | `"logit"` | none |
| `PROBIT` | `"probit"` | none |
| `AFFINE_LOGIT` | `"affine_logit"` | `low`, `high` |
| `AFFINE_PROBIT` | `"affine_probit"` | `low`, `high` |
| `LOWER_BOUNDED` | `"lower_bounded"` | `low` |
| `UPPER_BOUNDED` | `"upper_bounded"` | `high` |

Additional `TransformMethod` enum members currently not dispatched by `make_prior(...)`:

| __Enum Member__ | __String__ |
|:----------------|:----------:|
| `SIMPLEX` | `"simplex"` |
| `CHOLESKY_COV` | `"cholesky_cov"` |
| `CHOLESKY_CORR` | `"cholesky_corr"` |

## Example
```python
from SymbolicDSGE.bayesian import make_prior

priors = {
    "beta": make_prior(
        distribution="beta", # (1)!
        parameters={"a": 100.0, "b": 5.0, "loc": 0.0, "scale": 1.0},
        transform="logit", # (2)!
    ),
    "psi_pi": make_prior(
        distribution="normal",
        parameters={"mean": 2.0, "std": 0.5},
        transform="log", # (3)!
    ),
    "rho_g": make_prior(
        distribution="normal",
        parameters={"mean": 0.5, "std": 0.5},
        transform="affine_logit",
        transform_kwargs={"low": 0.0, "high": 1.0}, # (4)!
    ),
}
```

1. Parameter-space distribution on `(0, 1)` via beta family.
2. Maps `(0, 1)` into `(-inf, inf)` for unconstrained optimization.
3. Maps positive parameter space into `(-inf, inf)`.
4. Required kwargs for bounded affine transforms.
