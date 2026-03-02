---
tags:
    - doc
---
# LKJChol

Prior family over Cholesky factors of correlation matrices.

### Parameters
| __Argument__ | __Symbol__ | __Meaning__ | __Default__ |
|:-------------|:-----------|:------------|:-----------:|
| `eta` | $\eta$ | LKJ shape parameter (`eta > 0`) | `1.0` |
| `K` | $K$ | Correlation dimension | `-1` |
| `random_state` | - | RNG seed / generator | `None` |

???+ note "Dimension Parameter"
    `K` must be provided with a valid positive dimension for runtime use.

### PDF
$$
f(R;\eta) = C \times \lbrack \det{R} \rbrack^{\eta-1}
\\
\text{ }
\\
\text{ }
\\
C = 2^{\sum_{k=1}^{K-1} (2(\eta-1)+K-k)(K-k)} \times \prod_{k=1}^{K-1}\Bigg\lbrack \Beta\bigg(\eta + \frac{(K-k-1)}{2}, \eta + \frac{(K-k-1)}{2}\bigg)\Bigg\rbrack^{K-k}
$$

### Region
$$
L \in \mathcal{L}_K =
\left\{
L \in \mathbb{R}^{K\times K} :
L \text{ is lower triangular},\quad
L_{ii} > 0,\quad
\sum_{j=1}^{i} L_{ij}^2 = 1
\right\}
$$
