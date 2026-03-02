---
tags:
    - doc
---
# Normal

Standard Gaussian prior family.

### Parameters
| __Argument__ | __Symbol__ | __Meaning__ | __Default__ |
|:-------------|:-----------|:------------|:-----------:|
| `mean` | $\mu$ | Mean | `0.0` |
| `std` | $\sigma$ | Standard deviation | `1.0` |
| `random_state` | - | RNG seed / generator | `None` |

### PDF
$$
f(x;\mu,\sigma) =
\frac{1}{\sqrt{2\pi}\sigma}
\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

### Region
$$
x \in \mathbb{R}
$$
