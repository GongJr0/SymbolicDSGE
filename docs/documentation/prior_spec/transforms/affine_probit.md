---
tags:
    - doc
---
# AffineProbit

Bounded-interval transform to unconstrained space via affine scaling + probit.

### Parameters
| __Argument__ | __Meaning__ |
|:-------------|:------------|
| `low` | Lower bound of constrained interval |
| `high` | Upper bound of constrained interval |

### Transformation
$$
x \in (\text{low},\text{high}) \mapsto
y = \Phi^{-1}(z) \in \mathbb{R},
\quad
z = \frac{x-\text{low}}{\text{high}-\text{low}}
$$
