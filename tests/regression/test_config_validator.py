# type: ignore
import sympy as sp
import pytest

from SymbolicDSGE.regression.config import TemplateConfig
from SymbolicDSGE.regression.config_validator import (
    ConfigValidator,
    RestrictedBehaviorWarning,
)


def test_validate_config_accepts_defaults():
    cfg = TemplateConfig()
    out = ConfigValidator._validate_config(cfg)
    assert out == cfg


def test_validate_var_space_rejects_invalid_types():
    cfg = TemplateConfig(variable_space=["x", 1])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        ConfigValidator._validate_var_space(cfg)


def test_validate_hessian_restriction_rejects_unknown_mode():
    cfg = TemplateConfig(hessian_restriction="bad_mode")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hessian_restriction"):
        ConfigValidator._validate_hessian_restriction(cfg)


def test_validate_hessian_full_rejects_high_order_interactions():
    cfg = TemplateConfig(
        hessian_restriction="full",
        poly_interaction_order=2,
    )
    with pytest.raises(ValueError, match="Full Hessian restriction"):
        ConfigValidator._validate_hessian_restriction(cfg)


def test_validate_hessian_diag_warns_for_high_order_interactions():
    cfg = TemplateConfig(
        hessian_restriction="diag",
        poly_interaction_order=3,
    )
    with pytest.warns(RestrictedBehaviorWarning):
        ConfigValidator._validate_hessian_restriction(cfg)


def test_validate_per_parameter_complexity_bound_rejects_greater_than_model_bound():
    cfg = TemplateConfig(model_complexity_bound=4, per_parameter_complexity_bound=5)
    with pytest.raises(ValueError, match="cannot be greater"):
        ConfigValidator._validate_parameter_complexity_bound(cfg)


def test_validate_per_parameter_complexity_bound_warns_when_equal_to_model_bound():
    cfg = TemplateConfig(model_complexity_bound=4, per_parameter_complexity_bound=4)
    with pytest.warns(RestrictedBehaviorWarning):
        ConfigValidator._validate_parameter_complexity_bound(cfg)


def test_validate_interactions_warns_when_interaction_only_disabled():
    cfg = TemplateConfig(interaction_only=False)
    with pytest.warns(RestrictedBehaviorWarning):
        ConfigValidator._validate_interactions(cfg)


def test_validate_interactions_rejects_order_one_when_interaction_only_true():
    cfg = TemplateConfig(interaction_only=True, poly_interaction_order=1)
    with pytest.raises(ValueError, match="requires poly_interaction_order>1"):
        ConfigValidator._validate_interactions(cfg)


def test_validate_interaction_form_rejects_unknown_form():
    cfg = TemplateConfig(interaction_form="bad_form")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="interaction_form"):
        ConfigValidator._validate_interactions(cfg)


def test_validate_constant_filtering_rejects_unknown_strategy():
    cfg = TemplateConfig(constant_filtering="bad_strategy")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="constant filtering strategy"):
        ConfigValidator._validate_constant_handling(cfg)


def test_validate_constant_filtering_strip_warns():
    cfg = TemplateConfig(constant_filtering="strip")
    with pytest.warns(RestrictedBehaviorWarning):
        ConfigValidator._validate_constant_handling(cfg)


def test_validate_power_bounds_rejects_upper_lower_ordering():
    cfg = TemplateConfig(power_law_upper_bound=1, power_law_lower_bound=2)
    with pytest.raises(ValueError, match="cannot be less than"):
        ConfigValidator._validate_power_bound(cfg)


def test_validate_power_bounds_warns_on_unbounded_upper():
    cfg = TemplateConfig(power_law_upper_bound=None, power_law_lower_bound=0)
    with pytest.warns(UserWarning, match="allows for arbitrarily higher order"):
        ConfigValidator._validate_power_bound(cfg)


def test_validate_power_bounds_rejects_none_lower():
    cfg = TemplateConfig(power_law_upper_bound=2, power_law_lower_bound=None)
    with pytest.raises(ValueError, match="allows negative and fractional power laws"):
        ConfigValidator._validate_power_bound(cfg)


def test_validate_var_space_accepts_strings_symbols_and_functions():
    t = sp.Symbol("t", integer=True)
    cfg = TemplateConfig(variable_space=["x", sp.Symbol("y"), sp.Function("z")(t)])
    ConfigValidator._validate_var_space(cfg)
