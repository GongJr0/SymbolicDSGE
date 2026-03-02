---
tags:
    - doc
---
# Beta

Bounded prior family on the unit interval in its standard form.

### Parameters
| __Argument__ | __Symbol__ | __Meaning__ | __Default__ |
|:-------------|:-----------|:------------|:-----------:|
| `a` | $\alpha$ | First shape parameter | `1.0` |
| `b` | $\beta$ | Second shape parameter | `1.0` |
| `loc` | - | Lower shift used by implementation | `0.0` |
| `scale` | - | Width scale used by implementation | `1.0` |
| `random_state` | - | RNG seed / generator | `None` |

???+ note "Implementation Parameterization"
    The implementation follows `scipy.stats.beta(a, b, loc=loc, scale=scale)`, i.e. an affine-transformed beta distribution.

### PDF
$$
f(x;\alpha,\beta)=
\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}
x^{\alpha-1}(1-x)^{\beta-1}
$$

### Region
$$
x \in (0,1)
$$
