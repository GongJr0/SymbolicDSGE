---
tags:
    - doc
---
# Gamma

Positive prior family with shape-scale parameterization.

### Parameters
| __Argument__ | __Symbol__ | __Meaning__ | __Default__ |
|:-------------|:-----------|:------------|:-----------:|
| `a` | $k$ | Shape | `1.0` |
| `loc` | - | Location shift used by implementation | `0.0` |
| `scale` | $\theta$ | Scale | `1.0` |
| `random_state` | - | RNG seed / generator | `None` |

???+ note "Implementation Parameterization"
    The implementation follows `scipy.stats.gamma(a, loc=loc, scale=scale)`.

### PDF
$$
f(x;k,\theta)=
\frac{1}{\Gamma(k)\theta^k}
x^{k-1}e^{-x/\theta}
$$

### Region
$$
x \in (0,\infty)
$$
