from dataclasses import dataclass, asdict
from typing import Callable, Literal, Iterator, cast
from enum import StrEnum

import warnings

from pysr import ExpressionSpec, TemplateExpressionSpec, TensorBoardLoggerSpec
import sympy as sp


class OpType(StrEnum):
    BINARY = "binary"
    UNARY = "unary"


@dataclass(frozen=True)
class CustomOp:
    """
    Custom Operator Definition

    :param name: The name of the operator, used for mapping and constraints.
    :param type: The type of the operator, either binary or unary.
    :param lamb: A Python callable that implements the operator's functionality, used for sympy mapping.
    :param jl_str: The string representation of the operator in Julia, used for PySR configuration.
    :param complexity_bound: An optional complexity bound for the operator. For binary operators, this should be a tuple of two integers representing the maximum allowed complexities of the left and right operands. For unary operators, this should be a single integer representing the maximum allowed complexity of the operand. If not provided, no complexity constraints will be applied to the operator.
    :raises ValueError: If the complexity_bound is not of the correct type based on the operator type.
    :returns: None
    """

    name: str
    arity: OpType | Literal["binary", "unary"]
    lamb: Callable[..., float]
    jl_str: str
    primitive_operation: Callable[
        ..., sp.Expr
    ]  # Opetaror to substitute with callables to enable derivation of custom operators.
    complexity_bound: int | tuple[int, int] | None = None

    def _get_bound(self) -> int | tuple[int, int]:
        if self.complexity_bound is not None:
            return self.complexity_bound
        else:
            if self.arity == OpType.BINARY:
                return (-1, -1)
            elif self.arity == OpType.UNARY:
                return -1
            else:
                raise ValueError(
                    "Complexity bound must be a tuple of complexities or an integer, depending on operator type."
                )

    def _normalize_arity(self) -> None:
        if isinstance(self.arity, str):
            self.__setattr__("arity", OpType(self.arity))


# Buil-in Custom Operators:
def get_pow(p: int, prec: int) -> CustomOp:
    if p < 0:
        raise ValueError("Negative powers are not supported.")

    def pow(x: float) -> float:
        out: float = x**p
        return out

    def pow_primitive(x: sp.Expr) -> sp.Expr:
        return sp.Pow(x, p, evaluate=False)

    name = f"pow{prec}_{p}"
    jl_str = f"{name}(x) = Float{prec}(x^{p})"  # prec \in {16, 32, 64} (pre-validated)
    return CustomOp(
        name=name,
        arity=OpType.UNARY,
        lamb=pow,
        jl_str=jl_str,
        primitive_operation=pow_primitive,
        complexity_bound=-1,
    )


def get_sqrt(prec: int, eps: float = 1e-8) -> CustomOp:
    def ssqrt(x: float) -> float:
        out: float = (x**2 + eps) ** (
            0.25
        )  # sqrt approximation function robust to negative and 0 inputs.
        return out

    def ssqrt_primitive(x: sp.Expr) -> sp.Expr:
        return sp.sqrt(x, evaluate=False)

    name = f"ssqrt{prec}"
    jl_str = f"{name}(x) = Float{prec}((x*x + Float{prec}({eps}))^(0.25f0))"  # prec \in {16, 32, 64} (pre-validated)
    return CustomOp(
        name=name,
        arity=OpType.UNARY,
        lamb=ssqrt,
        jl_str=jl_str,
        primitive_operation=ssqrt_primitive,
        complexity_bound=-1,  # [default] Complexity of x is unrestricted
    )


def get_asinh(prec: int) -> CustomOp:
    def asinh(x: float) -> float:
        out: float = sp.asinh(x).evalf()  # pyright: ignore
        return out

    def asinh_primitive(x: sp.Expr) -> sp.Expr:
        return sp.asinh(x, evaluate=False)

    return CustomOp(
        name=f"asinh{prec}",
        arity=OpType.UNARY,
        lamb=asinh,
        jl_str=f"asinh{prec}(x) = Float{prec}(asinh(x))",  # asinh is part of the julia standard library
        primitive_operation=asinh_primitive,
        complexity_bound=-1,  # [default] Complexity of x is unrestricted
    )


def make_operator_general(
    lamb: Callable[..., float],
    jl_str: str,
    primitive_operation: Callable[..., sp.Expr],
    complexity_bound: tuple[int, int] | int | None = None,
) -> CustomOp:
    """
    Create a CustomOp for use in PySR.
    :param lamb: A Python function that implements the operator.
    :type lamb: Callable[..., float]
    :param jl_str: A string defining the operator in Julia syntax for PySR.
    :type jl_str: str
    :param primitive_operation: A sympy expression that defines the operator for use in sympy mapping and derivation.
    :type primitive_operation: Callable[..., sp.Expr]
    :param complexity_bound: An optional complexity bound for the operator.
    :type complexity_bound: tuple[int, int] | int | Non
    :return: A CustomOp object representing the operator.
    :rtype: CustomOp
    """
    # Parse name and arg count
    fun, _ = jl_str.split("=", maxsplit=1)
    name, args_str = fun.strip().split("(", maxsplit=1)

    name = name.strip()
    args_str = args_str.strip(")").strip()
    arg_count = len(args_str.split(","))

    if arg_count == 1:
        op_type = OpType.UNARY
        if (complexity_bound is not None) and not isinstance(complexity_bound, int):
            raise ValueError(
                f"Unary operators must have an integer complexity bound.\nGot: {complexity_bound}"
            )

    elif arg_count == 2:
        op_type = OpType.BINARY
        if (complexity_bound is not None) and not isinstance(complexity_bound, tuple):
            raise ValueError(
                f"Binary operators' complexity bounds must be a tuple of integers `(bound_arg1, bound_arg2)`.\nGot: {complexity_bound}"
            )
        if isinstance(complexity_bound, tuple) and len(complexity_bound) != 2:
            raise ValueError(
                f"Binary operators' complexity bounds must be a tuple of two integers `(bound_arg1, bound_arg2)`.\nGot: {complexity_bound}"
            )
    else:
        raise ValueError(
            f"Operators must be binary (2 arguments) or unary (1 argument).\nGot: {jl_str}"
        )

    return CustomOp(
        name=name,
        arity=op_type,
        lamb=lamb,
        jl_str=jl_str,
        primitive_operation=primitive_operation,
        complexity_bound=complexity_bound,
    )


@dataclass
class PySRParams:
    """
    Parameter Container with defaults for PySR symbolic regression. Some defaults are modified to better suit the DSGE context.
    """

    # Operator Definitions:
    binary_operators: list[str] | None = None
    unary_operators: list[str] | None = None
    extra_sympy_mappings: dict[str, Callable[..., float]] | None = None
    constraints: dict[str, int | tuple[int, int]] | None = None

    # Search Parameters:
    expression_spec: ExpressionSpec | TemplateExpressionSpec | None = None
    maxsize: int = 12
    maxdepth: int | None = None
    niterations: int = 100
    populations: int = 31
    population_size: int = 27
    ncycles_per_iteration: int = 380

    # Objective Function:
    elementwise_loss: str | None = None
    loss_function: str | None = None
    loss_function_expression: str | None = None
    loss_scale: Literal["log", "linear"] = "log"
    model_selection: Literal["accuracy", "best", "score"] = "best"
    dimensional_constraint_penalty: float = 1000.0
    dimensionless_constants_only: bool = False

    # Complexities:
    parsimony: float = 1e-3
    adaptive_parsimony_scaling: float = 1040.0
    nested_constraints: dict[str, dict[str, int]] | None = None
    complexity_of_operators: dict[str, int] | None = None
    complexity_of_constants: int = 0
    complexity_of_variables: int = 1
    complexity_mapping: None = None  # Not supported
    use_frequency: bool = True
    use_frequency_in_tournament: bool = True

    # Misc Parameters:
    warmup_maxsize_by: float = 0.0
    should_simplify: bool = True

    # Mutation Parameters:
    weight_add_node: float = 2.47
    weight_insert_node: float = 0.0112
    weight_delete_node: float = 0.870
    weight_do_nothing: float = 0.273
    weight_mutate_constant: float = 0.0346
    weight_mutate_operator: float = 0.293
    weight_swap_operands: float = 0.198
    weight_rotate_tree: float = 4.26
    weight_randomize: float = 0.000502
    weight_simplify: float = 0.00209
    weight_optimize: float = 0.0
    crossover_probability: float = 0.0259
    annealing: bool = False
    alpha: float = 3.17
    perturbation_factor: float = 0.129
    probability_negate_constant: float = 0.00743
    skip_mutation_failures: bool = True
    tournament_selection_n: int = 15
    tournament_selection_p: float = 0.982

    # Constant Optimization Parameters:
    optimizer_algorithm: Literal["BFGS", "NelderMead"] = "BFGS"
    optimizer_nrestarts: int = 2
    optimizer_f_calls_limit: int = 10000
    optimize_probability: float = 0.14
    optimizer_iterations: int = 8
    should_optimize_constants: bool = True

    # Migration Parameters:
    fraction_replaced: float = 0.00036
    fraction_replaced_hof: float = 0.0614
    migration: bool = True
    hof_migration: bool = True
    topn: int = 12

    # Data Preprocessing Parameters:
    denoise: bool = False
    select_k_features: int | None = None

    # Stopping Criteria:
    max_evals: int | None = None
    timeout_in_seconds: int | None = None
    early_stop_condition: str | None = None

    # Performance Optimization Parameters:
    parallelism: Literal["serial", "multithreading", "multiprocessing"] = (
        "multithreading"
    )
    procs: int | None = None
    cluster_manager: (
        Literal["slurm", "pbs", "lsf", "sge", "qrsh", "scyld", "htc"] | None
    ) = None
    heap_size_hint_in_bytes: int | None = None
    batching: bool = False
    batch_size: int = 50
    precision: int = 32  # 16f = 32c, 32f = 64c, 64f = 128c
    fast_cycle: bool = False
    turbo: bool = False
    bumper: bool = False
    autodiff_backend: Literal["Zygote"] | None = None  # None = forward-diff

    # Determinism Parameters:
    random_state: int | None = None
    deterministic: bool = False
    warm_start: bool = False

    # Monitoring and Logging Parameters:
    verbosity: int = 1
    update_verbosity: int | None = None
    print_precision: int = 5
    progress: bool = True
    logger_spec: TensorBoardLoggerSpec | None = None
    input_stream: Literal["stdin", "devnull"] = "stdin"

    # Environment Parameters:
    temp_equation_file: bool = True
    tempdir: str | None = None
    delete_tempfiles: bool = True
    update: bool = False

    # Exporting Parameters:
    output_directory: str | None = None
    run_id: int | None = None
    output_jax_format: bool = False  # Not supported
    output_torch_format: bool = False  # Not supported
    extra_jax_mappings: None = None  # Not supported
    extra_torch_mappings: None = None  # Not supported

    def add_operator(self, operator: CustomOp) -> None:
        if operator.arity == OpType.BINARY:
            if self.binary_operators is None:
                raise ValueError(
                    "Detected `self.binary_operators==None`. If you didn't manually set this please report it in the SymbolicDSGE GitHub."
                )

            self.binary_operators.append(operator.jl_str)
            if operator.complexity_bound is not None:
                if not isinstance(operator.complexity_bound, tuple):
                    raise ValueError(
                        "Binary operator complexity bound must be a tuple of two integers."
                    )
                if self.constraints is None:
                    self.constraints = {}
                self.constraints[operator.name] = operator.complexity_bound

        elif operator.arity == OpType.UNARY:
            if self.unary_operators is None:
                raise ValueError(
                    "Detected `self.unary_operators==None`. If you didn't manually set this please report it in the SymbolicDSGE GitHub."
                )

            self.unary_operators.append(operator.jl_str)
            if operator.complexity_bound is not None:
                if not isinstance(operator.complexity_bound, int):
                    raise ValueError(
                        "Unary operator complexity bound must be an integer."
                    )
                if self.constraints is None:
                    self.constraints = {}
                self.constraints[operator.name] = operator.complexity_bound

        else:
            raise ValueError("Invalid operator type.")

        if operator.lamb is not None:
            if self.extra_sympy_mappings is None:
                self.extra_sympy_mappings = {}
            self.extra_sympy_mappings[operator.name] = operator.lamb

            if self.constraints is None:
                self.constraints = {}
            self.constraints[operator.name] = operator._get_bound()
            ops = cast(list, self.unary_operators)
            for op2 in ops:
                self.set_nesting_constraint(operator.name, op2, 0)
                self.set_nesting_constraint(op2, operator.name, 0)

    def add_template(self, template_spec: TemplateExpressionSpec) -> None:
        self.expression_spec = template_spec

    def set_nesting_constraint(self, op1: str, op2: str, max_nesting: int) -> None:
        cleaned_ops = [
            op.split("=")[0].strip().split("(")[0].strip() if "=" in op else op
            for op in [op1, op2]
        ]  # must contain "=" if func def
        op1, op2 = cleaned_ops

        if self.nested_constraints is None:
            self.nested_constraints = {}
        if op1 not in self.nested_constraints:
            self.nested_constraints[op1] = {}
        self.nested_constraints[op1][op2] = max_nesting

    def _nesting_disable(self) -> dict[str, dict[str, int]]:
        if self.binary_operators is None:
            warnings.warn(
                "Detected `self.binary_operators==None`. If you didn't manually set this please report it in the SymbolicDSGE GitHub.",
                UserWarning,
            )
        if self.unary_operators is None:
            warnings.warn(
                "Detected `self.unary_operators==None`. If you didn't manually set this please report it in the SymbolicDSGE GitHub.",
                UserWarning,
            )

        ops = cast(list, self.unary_operators)

        # Clean function definitions from julia ops
        cleaned_ops = [
            op.split("=")[0].strip().split("(")[0].strip() if "=" in op else op
            for op in ops
        ]  # must contain "=" if func def
        return {op1: {op2: 0 for op2 in cleaned_ops} for op1 in cleaned_ops}

    def _default_binary_ops(self) -> list[str]:
        return ["+", "-", "*", "/"]

    def _default_unary_ops(self) -> list[str]:
        return []

    def _default_elementwise_loss(self) -> str:
        return "L2DistLoss()"

    def astict(self) -> dict[str, object]:
        return asdict(self)

    def __post_init__(self) -> None:
        if self.binary_operators is None:
            self.binary_operators = self._default_binary_ops()

        if self.unary_operators is None:
            self.unary_operators = self._default_unary_ops()

        if self.nested_constraints is None:
            self.nested_constraints = self._nesting_disable()

        if self.elementwise_loss is None:
            self.elementwise_loss = self._default_elementwise_loss()

    def __iter__(self) -> Iterator:
        return asdict(self).__iter__()

    def __getitem__(self, key: str) -> object:
        return asdict(self)[key]

    def keys(self) -> list[str]:
        return list(asdict(self).keys())


class ParameterCompatibilityError(Exception):
    """Custom exception for parameter compatibility issues with the template generator."""

    def __init__(self, msg: str) -> None:
        super().__init__(msg)


def validate_breaking_settings(params: PySRParams) -> PySRParams:
    """
    Validate parameters to ensure compatibility issues with the template generator are not present.
    :param params: A PySRParams object containing the parameters to validate.
    :type params: PySRParams
    :return: The validated PySRParams object.
    :rtype: PySRParams
    """
    # if params.complexity_of_constants != 0:
    #     raise ParameterCompatibilityError(
    #         (
    #             "Non-zero complexity of constants allows the constant terms to be at-or-above the complexity of variables. "
    #             "Functions that use complexity limits to disallow variable arguments will not work as intended. "
    #             "As of now, non-zero constant complexity is not supported but may be implemented in future versions.  "
    #         )
    #     )
    _MUST_BE_FALSE = {
        params.extra_torch_mappings,  # Expects None
        params.output_torch_format,  # Expects False
        params.output_jax_format,  # Expects False
        params.extra_jax_mappings,  # Expects None
        params.complexity_mapping,  # Expects None
    }

    if any(_MUST_BE_FALSE):
        raise ParameterCompatibilityError(
            (
                "One or more parameters are set to values that are not compatible with the template generator. "
                "The following parameters must be set to their default values: extra_torch_mappings (None), output_torch_format (False), output_jax_format (False), extra_jax_mappings (None), complexity_mapping (None). "
                "These features are not supported by the template generator and must be disabled for it to function properly. "
            )
        )

    if (params.parallelism != "serial") and (params.deterministic):
        raise ParameterCompatibilityError(
            (
                "Deterministic parallel execution is not supported by pysr (SR backend used). "
                "Please set parallelism to 'serial' or set deterministic to False to avoid this issue. "
            )
        )

    if (fp := params.precision) == 16:
        warnings.warn(
            (
                "16-bit precision is generally unadvisable as the whole julia backend (including solvers and GP evolution) will run in this precision setting. "
                "Unless runtime optiimization is a requirement, it is generally adivsable to keep precision at 32 or 64. "
                "NOTE: 16-bit precision can lead to overflows if the expected magnitudes of variables/constants/target are in the 10^5 range or above. "
                "This will not be caught in julia and can lead to undefined behavior via NaN/Inf at expression evaluations."
            ),
            UserWarning,
        )
    elif fp in (32, 64):
        pass
    else:
        raise ParameterCompatibilityError(
            (
                "Invalid precision value. Only 16, 32, and 64 are supported. "
                "Please set precision to one of these values to avoid compatibility issues with the template generator. "
            )
        )

    return params
