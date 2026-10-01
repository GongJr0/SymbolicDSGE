"""Transform dispatch module."""

from .transform import TransformMethod, Transform
from .identity import Identity
from .log import LogTransform
from .softplus import SoftplusTransform
from .logit import LogitTransform
from .probit import ProbitTransform
from .affine_logit import AffineLogitTransform
from .affine_probit import AffineProbitTransform
from .lower_bounded import LowerBoundedTransform
from .upper_bounded import UpperBoundedTransform
from .tanh import TanhTransform
from .cholesky_corr import CholeskyCorrTransform

TRANSFORM_METHOD_DISPATCH: dict[TransformMethod, type[Transform]] = {
    TransformMethod.IDENTITY: Identity,
    TransformMethod.LOG: LogTransform,
    TransformMethod.SOFTPLUS: SoftplusTransform,
    TransformMethod.LOGIT: LogitTransform,
    TransformMethod.PROBIT: ProbitTransform,
    TransformMethod.AFFINE_LOGIT: AffineLogitTransform,
    TransformMethod.AFFINE_PROBIT: AffineProbitTransform,
    TransformMethod.LOWER_BOUNDED: LowerBoundedTransform,
    TransformMethod.UPPER_BOUNDED: UpperBoundedTransform,
    TransformMethod.TANH: TanhTransform,
    TransformMethod.CHOLESKY_CORR: CholeskyCorrTransform,
}


def get_transform(method: str | TransformMethod | None) -> type[Transform]:
    """Get the transform class corresponding to the given method.

    Parameters
    ----------
    method : str | TransformMethod | None
        Method name of the transform. If None, returns the Identity transform.
        Must be a member of the :class:`TransformMethod` enum.

    Returns
    -------
    type[Transform]
        :class:`Transform` subclass corresponding to the given method.

    """
    if method is None:
        return Identity
    if method not in TRANSFORM_METHOD_DISPATCH:
        raise ValueError(
            f"Unsupported transform method: {method}\n please choose from: [{', '.join(TransformMethod)}]"
        )
    method_enum = TransformMethod(method)
    return TRANSFORM_METHOD_DISPATCH[method_enum]
