from SymbolicDSGE.regression.config import TemplateConfig
from .model_defaults import CustomOp, PySRParams, get_asinh, get_pow, get_sqrt


class BuiltInOpContainer:
    """
    Static class with methods accepting templates and generating SR operators accordingly.
    """

    @staticmethod
    def pow(config: TemplateConfig, params: PySRParams) -> CustomOp:
        upper = config.power_law_upper_bound or int(1e6)

        lower = config.power_law_lower_bound or 1
        prec = params.precision

        return get_pow(upper, lower, prec)

    @staticmethod
    def sqrt(params: PySRParams) -> CustomOp:
        prec = params.precision
        return get_sqrt(prec)

    @staticmethod
    def asinh() -> CustomOp:
        # built-in julia function inherits precision from the mdoel parameter. No need to set here.
        return get_asinh()
