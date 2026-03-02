---
tags:
    - doc
---
# HalfNormal

Non-negative prior family obtained by folding a normal distribution at zero.

### Parameters
| __Argument__ | __Symbol__ | __Meaning__ | __Default__ |
|:-------------|:-----------|:------------|:-----------:|
| `low` | - | Lower shift used by implementation | `0.0` |
| `scale` | $\sigma$ | Scale | `1.0` |
| `random_state` | - | RNG seed / generator | `None` |

???+ note "Implementation Parameterization"
    The implementation follows `scipy.stats.halfnorm(loc=low, scale=scale)`.

### PDF
$$
f(x;\sigma) =
\sqrt{\frac{2}{\pi\sigma^2}}
\exp\left(-\frac{x^2}{2\sigma^2}\right)
$$

### Region
$$
x \in [0,\infty)
$$
