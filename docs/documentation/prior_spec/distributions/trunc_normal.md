---
tags:
    - doc
---
# TruncNormal

Continuous normal prior truncated to a finite interval.

### Parameters
| __Argument__ | __Symbol__ | __Meaning__ | __Default__ |
|:-------------|:-----------|:------------|:-----------:|
| `low` | $\ell$ | Lower truncation bound | `-6.0` |
| `high` | $h$ | Upper truncation bound | `6.0` |
| `mean` | $\mu$ | Mean of the underlying normal distribution | `0.0` |
| `std` | $\sigma$ | Standard deviation of the underlying normal distribution | `1.0` |
| `random_state` | - | RNG seed / generator | `None` |

???+ note "Implementation Parameterization"
    This is a true truncated normal on the closed interval `[low, high]`.
    The density is the underlying normal renormalized over that interval;
    the implementation does not use clipped or censored point masses at the bounds.

### PDF
$$
f(x;\mu,\sigma,\ell,h)=
\frac{\phi\!\left(\frac{x-\mu}{\sigma}\right)}
{\sigma\left[\Phi\!\left(\frac{h-\mu}{\sigma}\right)-\Phi\!\left(\frac{\ell-\mu}{\sigma}\right)\right]}
$$

### Region
$$
x \in [\ell,h]
$$
