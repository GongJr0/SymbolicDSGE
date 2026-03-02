---
tags:
    - doc
---
# Uniform

Flat bounded prior family.

### Parameters
| __Argument__ | __Symbol__ | __Meaning__ | __Default__ |
|:-------------|:-----------|:------------|:-----------:|
| `low` | $a$ | Lower bound | `0.0` |
| `high` | $b$ | Upper bound | `1.0` |
| `random_state` | - | RNG seed / generator | `None` |

### PDF
$$
f(x;a,b)=\frac{1}{b-a}
$$

### Region
$$
x \in [a,b]
$$
