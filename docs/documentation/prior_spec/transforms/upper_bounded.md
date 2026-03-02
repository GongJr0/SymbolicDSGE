---
tags:
    - doc
---
# UpperBounded

Upper-bounded transform to unconstrained space.

### Parameters
| __Argument__ | __Meaning__ |
|:-------------|:------------|
| `high` | Upper bound of constrained interval |

### Transformation
$$
x \in (-\infty,\text{high}) \mapsto
y = \log(\text{high}-x) \in \mathbb{R}
$$
