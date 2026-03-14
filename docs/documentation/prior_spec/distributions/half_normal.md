---
tags:
    - doc
---
# HalfNormal

Non-negative prior family obtained by folding a normal distribution at zero.

### Parameters
| __Argument__ | __Symbol__ | __Meaning__ | __Default__ |
|:-------------|:-----------|:------------|:-----------:|
| `std` | $\sigma$ | Standard deviation of the folded underlying normal | `1.0` |
| `random_state` | - | RNG seed / generator | `None` |

???+ note "Implementation Parameterization"
    The implementation uses the canonical half-normal family on `[0, \infty)`.

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
