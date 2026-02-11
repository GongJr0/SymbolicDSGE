from dataclasses import dataclass
from typing import Literal
import sympy as sp


@dataclass(frozen=True)
class TemplateConfig:

    # Include Expression:
    # - True: include the expression in the template expression such that template = expr + template.
    # - False: template expression is only the template component without the initial expression.
    # This changes the implied target variable from deviations to levels.
    include_expression: bool = False

    # Variable Space:
    # - 'expr': Only include variables present in the initial expression.
    # - list[str | sp.Symbol | sp.Function]: Explicitly specify the variable space as a list of variable names or sympy symbols/functions.
    # - None: Allow all variables (unbounded variable space).
    variable_space: Literal["expr"] | list[str | sp.Symbol | sp.Function] | None = None

    # Hessian Restriction modes:
    # - Diag: Only enforce that the given (w.r.t) variable is linear in the expression.
    # - Full: Enfore completely affine expressions regardless of the (w.r.t.) variable.
    # Default: 'diag
    hessian_restriction: Literal["diag", "full"] | None = "diag"

    # Linearity Enforcement:
    # - True: Confirm a given template is linear in parameters (is expressable as a linear combination of parameters).
    # - False: Do not enforce linearity.
    # Default: False
    enforce_linearity: bool = False

    # Total Complexity Bound:
    # Enforce that the total complexity of the expression is bounded by a given integer C_glob.
    model_complexity_bound: int = 12

    # Per Parameter Complexity Bound:
    # On top of the global complexity bound, enforce a per param complexity c<C_glob for each linear term.
    # Asserts enforce_linearity = True when per_parameter_complexity_bound is not None.
    # Default: None
    per_parameter_complexity_bound: int | None = None

    # Power Law Order Bound:
    # Enforce that the order of power law terms (i.e. x^n) is bounded by a given integer n <= N.
    # Model defaults only allow x^2 by blocking general power laws and exposing a square(x) function.
    # Default: 2
    power_law_upper_bound: int | None = 2
    power_law_lower_bound: int | None = 1

    # Interaction  Only:
    # - True: Do not allow univariate linear components (\beta_i * x_i) in the expression.
    # - False: Allow all linear components.
    # Useful in ensuring pre-regression measurement coefficients do not change in simplification.
    # Example: y = 4x + 2 + f(x) where f(x) = 2x is simplified to y = x(4+2) + 2 = 6x + 2. This is not discovered structure, but a parameter re-estimation.
    # Default: True
    # Warns when set to False.
    interaction_only: bool = True

    # Polynomial Interaction Order:
    # Compute all combinatorial interactions up to the given order of polynomial terms.
    # For example, with variables [x1, x2] and poly_interaction_order = 2, the generated interactions are:
    # Degree 1: [x1, x2]
    # Degree 2: [x1*x1, x1*x2, x2*x2]
    # Default: 2
    poly_interaction_order: int = 2

    # Powers in Interactions:
    # - True: Allow power law terms (e.g. x^2) to be included in interactions.
    # - False: Only allow multiplicative interactions of variables (e.g. x1*x2).
    powers_in_interactions: bool = False

    # Interaction Form:
    # - 'func': Interactions take the form f(var1, var2, ...).
    # - 'prod': Interactions take the form f(var1*var2*...).
    interaction_form: Literal["func", "prod"] = "func"

    # Constant Filtering Strategy:
    # - 'disqualify': Disqualify expressions with constants.
    # - 'strip': Remove discovered constants.
    # - 'parametrize': Replace constants with parameters to be estimated.
    # - None: Do not apply any constant filtering.
    # Default: 'parametrize'
    constant_filtering: Literal["disqualify", "strip", "parametrize"] | None = (
        "parametrize"
    )
