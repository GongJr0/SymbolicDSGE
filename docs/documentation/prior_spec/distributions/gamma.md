---
tags:
    - doc
---
# Gamma

Positive prior family parameterized by mean and standard deviation.

### Parameters
| __Argument__ | __Symbol__ | __Meaning__ | __Default__ |
|:-------------|:-----------|:------------|:-----------:|
| `mean` | $\mu$ | Mean of the gamma random variable | `1.0` |
| `std` | $\sigma$ | Standard deviation of the gamma random variable | `1.0` |
| `random_state` | - | RNG seed / generator | `None` |

???+ note "Implementation Parameterization"
    The implementation converts `(mean, std)` into the gamma parameters `k` and `\theta`:

    $$
    k=\left(\frac{\mu}{\sigma}\right)^2,\quad
    \theta=\frac{\sigma^2}{\mu}
    $$

    These parameters are then used directly in the density and sampler.

### PDF
$$
f(x;k,\theta)=
\frac{1}{\Gamma(k)\theta^k}
x^{k-1}e^{-x/\theta}
$$

### Region
$$
x \in [0,\infty)
$$
