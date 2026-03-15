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
| `random_state` | - | RNG seed / generator | `None` |

???+ note "Implementation Parameterization"
    The implementation uses the canonical beta family on `[0, 1]`.
    The API only accepts the two shape parameters `a` and `b`;
    arbitrary `low` / `high` interval scaling is not part of this distribution class.

### PDF
$$
f(x;\alpha,\beta)=
\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}
x^{\alpha-1}(1-x)^{\beta-1}
$$

### Region
$$
x \in [0,1]
$$
