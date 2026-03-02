---
tags:
    - doc
---
# LowerBounded

Lower-bounded transform to unconstrained space.

### Parameters
| __Argument__ | __Meaning__ |
|:-------------|:------------|
| `low` | Lower bound of constrained interval |

### Transformation
$$
x \in (\text{low},\infty) \mapsto
y = \log(x-\text{low}) \in \mathbb{R}
$$
