from SymbolicDSGE.regression.config import TemplateConfig
from .model_defaults import CustomOp, PySRParams, get_asinh, get_pow, get_sqrt


class BuiltInOpContainer:
    """
    Static class with methods accepting templates and generating SR operators accordingly.
    """

    @staticmethod
    def pows(config: TemplateConfig, params: PySRParams) -> dict[str, CustomOp]:
        upper = config.power_law_upper_bound or 10

        lower = config.power_law_lower_bound or 1
        prec = params.precision
        out = {}
        for p in range(lower, upper + 1):
            out[f"pow{p}"] = get_pow(p, prec)
        return out

    @staticmethod
    def sqrt(params: PySRParams) -> CustomOp:
        prec = params.precision
        return get_sqrt(prec)

    @staticmethod
    def asinh(params: PySRParams) -> CustomOp:
        # built-in julia function inherits precision from the mdoel parameter. No need to set here.
        prec = params.precision
        return get_asinh(prec)
