---
tags:
    - doc
---
# InvGamma

Inverse-gamma prior family.

### Parameters
| __Argument__ | __Symbol__ | __Meaning__ | __Default__ |
|:-------------|:-----------|:------------|:-----------:|
| `mean` | $\mu$ | Mean of the inverse-gamma prior | `1.0` |
| `std` | $\sigma$ | Standard deviation of the inverse-gamma prior | `1.0` |
| `random_state` | - | RNG seed / generator | `None` |

???+ note "Implementation Parameterization"
    The implementation derives the canonical inverse-gamma parameters `\alpha` and `\beta` from `mean` and `std`:
    `#!python alpha = 2 + (mean / std) ** 2` and
    `#!python beta = mean * (alpha - 1)`.
    The public API does not expose separate `loc` / `scale` wrapper parameters.

### PDF
$$
f(x;\alpha,\beta)=
\frac{\beta^\alpha}{\Gamma(\alpha)}
x^{-(\alpha+1)}
\exp\left(-\frac{\beta}{x}\right)
$$

### Region
$$
x \in (0,\infty)
$$
