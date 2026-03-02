---
tags:
    - doc
---
# HalfCauchy

Heavy-tailed positive prior family.

### Parameters
| __Argument__ | __Symbol__ | __Meaning__ | __Default__ |
|:-------------|:-----------|:------------|:-----------:|
| `low` | - | Lower shift used by implementation | `0.0` |
| `scale` | $\gamma$ | Scale | `1.0` |
| `random_state` | - | RNG seed / generator | `None` |

???+ note "Implementation Parameterization"
    The implementation follows `scipy.stats.halfcauchy(loc=low, scale=scale)`.

### PDF
$$
f(x;\gamma) =
\frac{2}{\pi\gamma\left(1 + (x/\gamma)^2\right)}
$$

### Region
$$
x \in [0,\infty)
$$
