---
tags:
    - doc
---
# HalfCauchy

Heavy-tailed positive prior family.

### Parameters
| __Argument__ | __Symbol__ | __Meaning__ | __Default__ |
|:-------------|:-----------|:------------|:-----------:|
| `gamma` | $\gamma$ | Half-Cauchy scale parameter | `1.0` |
| `random_state` | - | RNG seed / generator | `None` |

???+ note "Implementation Parameterization"
    The implementation uses the canonical half-Cauchy family on `[0, \infty)`.
    No shift parameter is exposed.

### PDF
$$
f(x;\gamma) =
\frac{2}{\pi\gamma\left(1 + \left(\frac{x}{\gamma}\right)^2\right)}
$$

### Region
$$
x \in [0,\infty)
$$
