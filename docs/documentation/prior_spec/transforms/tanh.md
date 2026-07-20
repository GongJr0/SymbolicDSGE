---
tags:
    - doc
---
# Tanh

Correlation-space transform mapping the open interval $(-1, 1)$ to unconstrained space. Used for standalone correlation parameters, such as an isolated off-diagonal entry of a `Q` or `R` matrix.

### Parameters
This transform has no parameters.

### Transformation
$$
x \in (-1,1) \mapsto y = \tanh^{-1}(x) \in \mathbb{R}
$$

$$
y \in \mathbb{R} \mapsto x = \tanh(y) \in (-1,1)
$$
