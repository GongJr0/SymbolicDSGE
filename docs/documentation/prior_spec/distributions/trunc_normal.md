---
tags:
    - doc
---
# TruncNormal

Normal prior truncated to a finite interval.

### Parameters
| __Argument__ | __Symbol__ | __Meaning__ | __Default__ |
|:-------------|:-----------|:------------|:-----------:|
| `low` | $\ell$ | Lower truncation bound | `-6.0` |
| `high` | $h$ | Upper truncation bound | `6.0` |
| `loc` | $\mu$ | Location | `0.0` |
| `scale` | $\sigma$ | Scale | `1.0` |
| `random_state` | - | RNG seed / generator | `None` |

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
