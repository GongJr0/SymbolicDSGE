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
}


def get_transform(method: str) -> type[Transform]:
    if method not in TRANSFORM_METHOD_DISPATCH:
        raise ValueError(
            f"Unsupported transform method: {method}\n please choose from: {list(TRANSFORM_METHOD_DISPATCH.values())}"
        )
    method_enum = TransformMethod(method)
    return TRANSFORM_METHOD_DISPATCH[method_enum]
