---
tags:
    - doc
---
# AffineLogit

Bounded-interval transform to unconstrained space via affine scaling + logit.

### Parameters
| __Argument__ | __Meaning__ |
|:-------------|:------------|
| `low` | Lower bound of constrained interval |
| `high` | Upper bound of constrained interval |

### Transformation
$$
x \in (\text{low},\text{high}) \mapsto
y = \log\left(\frac{z}{1-z}\right) \in \mathbb{R},
\quad
z = \frac{x-\text{low}}{\text{high}-\text{low}}
$$
