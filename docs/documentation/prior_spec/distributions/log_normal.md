---
tags:
    - doc
---
# LogNormal

Positive-support prior family based on exponentiated normals.

### Parameters
| __Argument__ | __Symbol__ | __Meaning__ | __Default__ |
|:-------------|:-----------|:------------|:-----------:|
| `s` | $\sigma$ | Shape (std of underlying normal) | `1.0` |
| `low` | - | Lower shift used by implementation | `0.0` |
| `scale` | - | Scale used by implementation | `1.0` |
| `random_state` | - | RNG seed / generator | `None` |

???+ note "Implementation Parameterization"
    The implementation follows `scipy.stats.lognorm(s, loc=low, scale=scale)`. The PDF above is the standard unshifted form.

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
