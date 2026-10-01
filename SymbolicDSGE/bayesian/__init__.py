"""Prior components for Bayesian estimation of DSGE models.

Contains distributions, transforms, and relevant utilities to construct complete prior objects
with bounded distributions mapped to unconstrained sampling spaces. Distributions and transforms are
independently usable, but the public surface routes through the :func:`make_prior` only.

:class:`DistributionFamily` and :class:`TransformMethod` name the strings
``make_prior`` accepts. Both are ``StrEnum``, so a member and its own value are
interchangeable wherever one is asked for.
"""

from .distributions.distribution import DistributionFamily
from .priors import Prior, make_prior
from .transforms.transform import TransformMethod

__all__ = [
    "DistributionFamily",
    "Prior",
    "TransformMethod",
    "make_prior",
]
