---
tags:
    - doc
---
# InvGamma

Inverse-gamma prior family.

### Parameters
| __Argument__ | __Symbol__ | __Meaning__ | __Default__ |
|:-------------|:-----------|:------------|:-----------:|
| `a` | $\alpha$ | Shape | `1.0` |
| `loc` | - | Location shift used by implementation | `0.0` |
| `scale` | $\beta$ | Scale | `1.0` |
| `random_state` | - | RNG seed / generator | `None` |

???+ note "Implementation Parameterization"
    The implementation follows `scipy.stats.invgamma(a, loc=loc, scale=scale)`.

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
