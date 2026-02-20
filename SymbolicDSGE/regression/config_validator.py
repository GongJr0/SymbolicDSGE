from .config import TemplateConfig, HessianMode, InteractionForm, ConstantFiltering
import sympy as sp
import warnings


class RestrictedBehaviorWarning(UserWarning):
    """Warns when config settings create restricted behavior that may not be intended."""

    def __init__(self, message: str):
        super().__init__(message)


class ConfigValidator:
    @staticmethod
    def _validate_config(config: TemplateConfig) -> TemplateConfig:
        """
        Check config fields' consistency and validity.

        :param config: Configuration to validate.
        :type config: TemplateConfig
        :return: Validated configuration.
        :rtype: TemplateConfig
        """
        ConfigValidator._validate_var_space(config)
        # ConfigValidator._validate_poly_interaction_order(config)
        ConfigValidator._validate_hessian_restriction(config)
        ConfigValidator._validate_parameter_complexity_bound(config)
        ConfigValidator._validate_interactions(config)
        ConfigValidator._validate_constant_handling(config)
        ConfigValidator._validate_power_bound(config)

        return config

    @staticmethod
    def _validate_var_space(config: TemplateConfig) -> None:
        if config.variable_space is not None:
            if not all(
                isinstance(var, (str, sp.Symbol, sp.Function))
                for var in config.variable_space
            ):
                raise TypeError(
                    "All elements in variable_space must be of type str, sympy.Symbol, or sympy.Function."
                )

    @staticmethod
    def _validate_poly_interaction_order(config: TemplateConfig) -> None:
        if not isinstance((order := config.poly_interaction_order), int):
            raise TypeError("poly_interaction_order must be an integer.")
        elif order < 1:
            raise ValueError("poly_interaction_order must be a positive integer.")

    @staticmethod
    def _validate_hessian_restriction(config: TemplateConfig) -> None:
        if (hessian := config.hessian_restriction) not in HessianMode:
            raise ValueError(
                "hessian_restriction must be one of 'free', 'diag', or 'full'."
            )
        elif hessian == "full" and config.poly_interaction_order > 1:
            raise ValueError(
                (
                    "Full Hessian restriction creates affine models where all variables are "
                    "isolated and are of the first order."
                )
            )
        elif hessian == "diag" and config.poly_interaction_order > 2:
            warnings.warn(
                (
                    f"{config.poly_interaction_order=} has been specified with diagonal Hessian restriction. "
                    "Diagonal restriction disallows interactions of order>1. "
                    "If you wish to include higher order interactions, please set `hessian_restriction=None`"
                ),
                RestrictedBehaviorWarning,
            )

    @staticmethod
    def _validate_parameter_complexity_bound(config: TemplateConfig) -> None:
        if (pbound := config.per_parameter_complexity_bound) is not None:
            if pbound > config.model_complexity_bound:
                raise ValueError(
                    (
                        f"per_parameter_complexity_bound={pbound} cannot be greater than {config.model_complexity_bound=}. "
                        "Please ensure per_parameter_complexity_bound <= model_complexity_bound."
                    )
                )
            elif pbound == config.model_complexity_bound:
                warnings.warn(
                    (
                        f"per_parameter_complexity_bound={pbound} is equal to model_complexity_bound={config.model_complexity_bound}. "
                        "This will result on a single parameter term being generated. "
                        "If you want to ensure all interaction terms are included, "
                        "consider increasing model_complexity_bound or decreasing per_parameter_complexity_bound."
                    ),
                    RestrictedBehaviorWarning,
                )

    @staticmethod
    def _validate_interactions(config: TemplateConfig) -> None:
        # Interaction and Transformation Only Block
        if not (int_policy := config.interaction_only):
            warnings.warn(
                (
                    f"{config.interaction_only=} allows univariate, non-transformed linear components in the generated template. "
                    "If the initial expression includes linear terms, the generated template can be simplified into the same functional form with different coefficients. "
                ),
                RestrictedBehaviorWarning,
            )
        elif int_policy and config.poly_interaction_order == 1:
            raise ValueError(
                (
                    f"{config.interaction_only=} requires poly_interaction_order>1 as combinations of order 1 will produce variables themselves,"
                )
            )

        # Interaction Format Block
        if config.interaction_form not in InteractionForm:
            raise ValueError("interaction_form must be either 'func' or 'prod'.")

    @staticmethod
    def _validate_constant_handling(config: TemplateConfig) -> None:
        if (const_strategy := config.constant_filtering) not in ConstantFiltering:
            raise ValueError(
                (
                    f"constant_filtering= is not a valid constant filtering strategy. "
                    "Please set constant_filtering to one of None, 'disqualify', 'strip', or 'parametrize'."
                )
            )
        elif const_strategy == "strip":
            warnings.warn(
                (
                    "constant_filtering='strip' will remove constants from the model discovered expressions. "
                    "This is strongly discouraged and strategies such as 'disqualify' can restrict the model to avoid constants. "
                    "If you want to allow constants in discovery as parameters, you can use 'parametrize' and later estimate the constants yourself."
                ),
                RestrictedBehaviorWarning,
            )

    @staticmethod
    def _validate_power_bound(config: TemplateConfig) -> None:
        if (config.power_law_upper_bound is not None) and (
            config.power_law_lower_bound is not None
        ):
            if config.power_law_upper_bound < config.power_law_lower_bound:
                raise ValueError(
                    (
                        f"power_law_upper_bound= cannot be less than power_law_lower_bound={config.power_law_lower_bound}. "
                        "Please ensure power_law_upper_bound >= power_law_lower_bound."
                    )
                )

        if (pow_upper := config.power_law_upper_bound) is None:
            warnings.warn(
                (
                    f"{pow_upper=} allows for arbitrarily higher order power law terms. "
                    "This lets the model create any polynomial term within the complexity bounds and can lead to overfitting very easily."
                    "Consider setting a finite power_law_upper_bound to restrict the search space."
                ),
                UserWarning,
            )
        if (pow_lower := config.power_law_lower_bound) is None:
            raise ValueError(
                (
                    f"{pow_lower=} allows negative and fractional power laws. "
                    "both negative power laws and freeform roots (fractional power) are not supported. "
                    "Please set a non-negative integer power_law_lower_bound."
                )
            )
