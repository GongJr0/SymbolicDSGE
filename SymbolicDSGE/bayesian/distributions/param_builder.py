from .norm import NORM_DEFAULTS
from .log_norm import LOGNORM_DEFAULTS
from .half_norm import HALFNORM_DEFAULTS
from .trunc_norm import TRUNCNORM_DEFAULTS
from .beta import BETA_DEFAULTS
from .gamma import GAMMA_DEFAULTS
from .inv_gamma import INVGAMMA_DEFAULTS
from .half_cauchy import HALF_CAUCHY_DEFAULTS
from .lkj_chol import LKJ_DEFAULTS
from .uniform import UNIFORM_DEFAULTS
from .distribution import DistributionFamily

from typing import Any, Mapping, cast

DIST_PARAMS_DISPATCH: dict[DistributionFamily, Mapping[str, Any]] = {
    DistributionFamily.NORMAL: NORM_DEFAULTS,
    DistributionFamily.LOGNORMAL: LOGNORM_DEFAULTS,
    DistributionFamily.HALFNORMAL: HALFNORM_DEFAULTS,
    DistributionFamily.TRUNCNORMAL: TRUNCNORM_DEFAULTS,
    DistributionFamily.BETA: BETA_DEFAULTS,
    DistributionFamily.GAMMA: GAMMA_DEFAULTS,
    DistributionFamily.INVGAMMA: INVGAMMA_DEFAULTS,
    DistributionFamily.HALFCAUCHY: HALF_CAUCHY_DEFAULTS,
    DistributionFamily.LKJCHOL: LKJ_DEFAULTS,
    DistributionFamily.UNIFORM: UNIFORM_DEFAULTS,
}


def get_dist_params(family: str) -> dict[str, Any]:
    if family not in DIST_PARAMS_DISPATCH:
        raise ValueError(
            f"Unsupported distribution family: {family}\n please choose from: {list(DIST_PARAMS_DISPATCH.keys())}"
        )
    family_enum = DistributionFamily(family)
    return cast(dict, DIST_PARAMS_DISPATCH[family_enum])
