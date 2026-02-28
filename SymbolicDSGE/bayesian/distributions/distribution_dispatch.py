from .distribution import Distribution, DistributionFamily
from .norm import Normal
from .log_norm import LogNormal
from .half_norm import HalfNormal
from .trunc_norm import TruncNormal
from .half_cauchy import HalfCauchy
from .beta import Beta
from .gamma import Gamma
from .inv_gamma import InvGamma
from .lkj_chol import LKJChol
from .uniform import Uniform

DISTRIBUTION_DISPATCH: dict[DistributionFamily, type[Distribution]] = {
    DistributionFamily.NORMAL: Normal,
    DistributionFamily.LOGNORMAL: LogNormal,
    DistributionFamily.HALFNORMAL: HalfNormal,
    DistributionFamily.TRUNCNORMAL: TruncNormal,
    DistributionFamily.HALFCAUCHY: HalfCauchy,
    DistributionFamily.BETA: Beta,
    DistributionFamily.GAMMA: Gamma,
    DistributionFamily.INVGAMMA: InvGamma,
    DistributionFamily.UNIFORM: Uniform,
    DistributionFamily.LKJCHOL: LKJChol,
}


def get_distribution(family: str) -> type[Distribution]:
    if family not in DISTRIBUTION_DISPATCH:
        raise ValueError(
            f"Unsupported distribution family: {family}\n please choose from: {list(DISTRIBUTION_DISPATCH.values())}"
        )
    family_enum = DistributionFamily(family)
    return DISTRIBUTION_DISPATCH[family_enum]
