# type: ignore
import pytest
import sympy as sp

from SymbolicDSGE.regression.built_in_op_container import BuiltInOpContainer
from SymbolicDSGE.regression.config import TemplateConfig
from SymbolicDSGE.regression.model_defaults import (
    CustomOp,
    OpType,
    ParameterCompatibilityError,
    PySRParams,
    get_pow,
    make_operator_general,
    validate_breaking_settings,
)


def test_get_pow_rejects_negative_powers():
    with pytest.raises(ValueError, match="Negative powers"):
        get_pow(-1, 32)


def test_make_operator_general_detects_arity_and_bounds():
    unary = make_operator_general(
        lamb=lambda x: x,
        jl_str="foo(x) = x",
        primitive_operation=lambda x: x,
        complexity_bound=2,
    )
    assert unary.arity == OpType.UNARY
    assert unary.name == "foo"

    binary = make_operator_general(
        lamb=lambda x, y: x + y,
        jl_str="bar(x, y) = x + y",
        primitive_operation=lambda x, y: x + y,
        complexity_bound=(1, 2),
    )
    assert binary.arity == OpType.BINARY
    assert binary.name == "bar"


def test_make_operator_general_rejects_invalid_complexity_type():
    with pytest.raises(ValueError, match="Unary operators must have an integer"):
        make_operator_general(
            lamb=lambda x: x,
            jl_str="foo(x) = x",
            primitive_operation=lambda x: x,
            complexity_bound=(1, 2),  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="must be a tuple"):
        make_operator_general(
            lamb=lambda x, y: x + y,
            jl_str="bar(x, y) = x + y",
            primitive_operation=lambda x, y: x + y,
            complexity_bound=3,  # type: ignore[arg-type]
        )


def test_make_operator_general_rejects_non_unary_non_binary():
    with pytest.raises(ValueError, match="must be binary"):
        make_operator_general(
            lamb=lambda x, y, z: x + y + z,
            jl_str="baz(x, y, z) = x + y + z",
            primitive_operation=lambda x, y, z: x + y + z,
        )


def test_pysrparams_post_init_sets_defaults():
    params = PySRParams()
    assert params.binary_operators == ["+", "-", "*", "/"]
    assert params.unary_operators == []
    assert params.elementwise_loss == "L2DistLoss()"
    assert params.nested_constraints == {}


def test_pysrparams_add_unary_operator_updates_params_and_constraints():
    params = PySRParams()
    op = make_operator_general(
        lamb=lambda x: x,
        jl_str="foo(x) = x",
        primitive_operation=lambda x: x,
        complexity_bound=2,
    )
    params.add_operator(op)

    assert any("foo(" in s for s in params.unary_operators or [])
    assert params.constraints is not None and params.constraints["foo"] == 2
    assert (
        params.extra_sympy_mappings is not None and "foo" in params.extra_sympy_mappings
    )
    assert (
        params.nested_constraints is not None
        and params.nested_constraints["foo"]["foo"] == 0
    )


def test_pysrparams_add_binary_operator_registers_tuple_constraint():
    params = PySRParams()
    op = make_operator_general(
        lamb=lambda x, y: x + y,
        jl_str="bar(x, y) = x + y",
        primitive_operation=lambda x, y: x + y,
        complexity_bound=(1, 2),
    )
    params.add_operator(op)

    assert any("bar(" in s for s in params.binary_operators or [])
    assert params.constraints is not None and params.constraints["bar"] == (1, 2)


def test_pysrparams_add_operator_rejects_invalid_bound_type_for_arity():
    params = PySRParams()
    bad = CustomOp(
        name="bad",
        arity=OpType.UNARY,
        lamb=lambda x: x,
        jl_str="bad(x) = x",
        primitive_operation=lambda x: x,
        complexity_bound=(1, 2),  # type: ignore[arg-type]
    )
    with pytest.raises(
        ValueError, match="Unary operator complexity bound must be an integer"
    ):
        params.add_operator(bad)


def test_set_nesting_constraint_cleans_operator_names():
    params = PySRParams()
    params.set_nesting_constraint("foo(x) = x", "bar(x) = x", 1)
    assert params.nested_constraints is not None
    assert params.nested_constraints["foo"]["bar"] == 1


def test_validate_breaking_settings_rejects_unsupported_mapping_settings():
    params = PySRParams(output_torch_format=True)
    with pytest.raises(ParameterCompatibilityError, match="not compatible"):
        validate_breaking_settings(params)


def test_validate_breaking_settings_rejects_deterministic_parallel():
    params = PySRParams(parallelism="multithreading", deterministic=True)
    with pytest.raises(
        ParameterCompatibilityError, match="Deterministic parallel execution"
    ):
        validate_breaking_settings(params)


def test_validate_breaking_settings_warns_on_16bit_precision():
    params = PySRParams(precision=16)
    with pytest.warns(UserWarning, match="16-bit precision"):
        out = validate_breaking_settings(params)
    assert out is params


def test_validate_breaking_settings_rejects_invalid_precision():
    params = PySRParams(precision=8)  # type: ignore[arg-type]
    with pytest.raises(ParameterCompatibilityError, match="Invalid precision value"):
        validate_breaking_settings(params)


def test_validate_breaking_settings_accepts_supported_defaults():
    params = PySRParams(precision=32, parallelism="serial", deterministic=True)
    assert validate_breaking_settings(params) is params


def test_built_in_op_container_exposes_power_sqrt_and_asinh_ops():
    cfg = TemplateConfig(power_law_lower_bound=2, power_law_upper_bound=3)
    params = PySRParams(precision=32)

    pows = BuiltInOpContainer.pows(cfg, params)
    assert set(pows.keys()) == {"pow2", "pow3"}
    assert all(op.arity == OpType.UNARY for op in pows.values())

    sqrt_op = BuiltInOpContainer.sqrt(params)
    asinh_op = BuiltInOpContainer.asinh(params)
    assert sqrt_op.name.startswith("ssqrt")
    assert asinh_op.name.startswith("asinh")
