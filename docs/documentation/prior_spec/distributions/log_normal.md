---
tags:
    - doc
---
# LogNormal

Positive-support prior family based on exponentiated normals.

### Parameters
| __Argument__ | __Symbol__ | __Meaning__ | __Default__ |
|:-------------|:-----------|:------------|:-----------:|
| `mean` | $\mu$ | Mean of the underlying normal distribution | `0.0` |
| `std` | $\sigma$ | Standard deviation of the underlying normal distribution | `1.0` |
| `random_state` | - | RNG seed / generator | `None` |

???+ note "Implementation Parameterization"
    The implementation uses the underlying normal parameters directly:
    `#!python X = exp(N(mean, std^2))`.

### PDF
$$
f(x;\mu,\sigma) =
\frac{1}{x\sigma\sqrt{2\pi}}
\exp\left(-\frac{(\log x-\mu)^2}{2\sigma^2}\right)
$$

### Region
$$
x \in (0,\infty)
$$
