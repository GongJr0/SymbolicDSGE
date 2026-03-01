from __future__ import annotations
import warnings

from .transforms.transform import TransformMethod, Transform
from .transforms.transform_dispatch import get_transform

from .distributions.distribution import Distribution, RandomState, Size, VecF64
from .distributions.distribution_dispatch import get_distribution
from .distributions.param_builder import get_dist_params

from .support import Support, bounded

from typing import TypedDict, Any, cast
from numpy import float64
from numpy.typing import NDArray


from dataclasses import dataclass
from typing import overload


class PriorDispatch(TypedDict):
    distribution: Distribution
    parameters: dict[str, Any]  # TypedDicts for each family
    transform: TransformMethod
    transform_kwargs: dict[str, Any] | None  # TypedDicts for each transform


@dataclass(frozen=True)
class Prior:
    dist: Distribution
    transform: Transform

    @overload
    def logpdf(self, z: float64) -> float64: ...
    @overload
    def logpdf(self, z: NDArray[float64]) -> NDArray[float64]: ...

    @bounded(domain="maps_to")
    def logpdf(self, z: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        x = self.transform.inverse(z)
        return self.dist.logpdf(x) + self.transform.log_det_abs_jacobian_inverse(z)

    @overload
    def grad_logpdf(self, x: float64) -> float64: ...
    @overload
    def grad_logpdf(self, x: NDArray[float64]) -> NDArray[float64]: ...

    @bounded(domain="maps_to")
    def grad_logpdf(self, z: float64 | NDArray[float64]) -> float64 | NDArray[float64]:
        x = self.transform.inverse(z)
        gx = self.dist.grad_logpdf(x)
        dx_dz = self.transform.grad_inverse(z)
        return dx_dz * gx + self.transform.grad_log_det_abs_jacobian_inverse(z)

    def rvs(self, size: Size, random_state: RandomState) -> NDArray[float64]:
        return cast(VecF64, self.dist.rvs(size, random_state))

    def _confirm_bound_match(self) -> None:
        _sup = self.dist.support
        _map = self.transform.maps_to
        _trans_sup = self.transform.support
        if _sup != _map:
            raise ValueError(
                f"Distribution support {self.dist.support} does not match transform maps_to {self.transform.maps_to}"
            )

        if _sup != _trans_sup:
            raise ValueError(
                "The transform's support function must match the distribution's support when using a ConstrainedPrior. "
                "ConstrainedPrior assumes the distribution is already constrained to the desired support; "
                "Transformations will not use jacobian corrections. "
            )

        if not self.dist.support.is_finite:
            warnings.warn(
                "ConstrainedPrior created with non-finite support. "
                "This class assumes a pre-constrained distribution and the transformation will not be applied with jacobian correction. "
                "If this is intentional, you can ignore this warning. Otherwise, consider using an unconstrained distribution with a transformation that maps to the desired support."
            )

    @property
    def support(self) -> Support:
        return self.dist.support

    @property
    def maps_to(self) -> Support:
        return self.transform.maps_to


def make_prior(
    distribution: str,
    parameters: dict[str, Any],
    transform: str,
    transform_kwargs: dict[str, Any] | None = None,
) -> Prior:
    dist = get_distribution(distribution)
    _transform = get_transform(transform)
    param_dict = get_dist_params(distribution)

    if not all(param in param_dict for param in parameters):
        raise ValueError(
            f"Unrecognized parameters for distribution {distribution}. Expected parameters: {param_dict.keys()}. Received parameters: {parameters.keys()}"
        )
    params_override = param_dict | parameters
    dist_inst = dist(**params_override)
    transform_inst = _transform(**(transform_kwargs or {}))

    return Prior(dist=dist_inst, transform=transform_inst)
