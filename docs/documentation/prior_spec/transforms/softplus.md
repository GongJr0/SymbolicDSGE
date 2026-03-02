---
tags:
    - doc
---
# Softplus

Smooth positive-space transform used as an alternative to `log`.

### Parameters
This transform has no parameters.

### Transformation
$$
x \in (0,\infty) \mapsto y = \log(e^x - 1) \in \mathbb{R}
$$

$$
y \in \mathbb{R} \mapsto x = \log(1 + e^y) \in (0,\infty)
$$
