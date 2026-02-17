# type: ignore
# %%
from SymbolicDSGE import ModelParser, DSGESolver, Shock
import sympy as sp
from warnings import catch_warnings, simplefilter
import numpy as np

from SymbolicDSGE.regression import (
    BaseModelParametrizer,
    PySRParams,
    TemplateConfig,
    SymbolicRegressor,
)

# %%
conf, kalman = ModelParser("MODELS/POST82.yaml").get_all()

with catch_warnings():
    simplefilter(action="ignore")
    mat = sp.Matrix(conf.equations.model)
mat

# %%
sol = DSGESolver(conf, kalman)
comp = sol.compile(variable_order=conf.variables, n_state=3, n_exog=2)
conf.equations.observable

# %%
solved = sol.solve(
    comp,
    steady_state=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
    log_linear=False,
)

# %%
template_conf = TemplateConfig(
    include_expression=True,
    poly_interaction_order=2,
    interaction_form="func",
    hessian_restriction="free",
)
params = PySRParams(
    precision=32,
    maxsize=12,
    niterations=100,
    complexity_of_constants=3,
    should_optimize_constants=False,
    should_simplify=False,
)
parametrizer = BaseModelParametrizer(
    variable_names=conf.variables[comp.n_exog :], params=params, config=template_conf
)

parametrizer.add_built_in_ops()
parametrizer.make_and_add_template(
    expr=conf.equations.observable["Infl"].subs(conf.calibration.parameters)
)

sr = SymbolicRegressor(parametrizer=parametrizer)
parametrizer.built_in_ops
# %%
T = 200
g_shock = Shock(T, "norm", seed=0, dist_kwargs={"loc": 0.0}).shock_generator()
z_shock = Shock(T, "norm", seed=1, dist_kwargs={"loc": 0.0}).shock_generator()

sim_shocks = {"g": g_shock, "z": z_shock}

# sim_shocks = np.array([[1.0, 1.0], *np.zeros((24, 2))])
sol = solved.sim(T, sim_shocks, observables=True)

# %%
# SR Test
X = np.column_stack([sol[var] for var in comp.var_names[comp.n_exog :]])
X = np.asarray(X, dtype=np.float32)

y = sol["Infl"]
var_names = conf.variables[2:]
sr.model.expression_spec.combine
expr = sr.fit(X, y, variable_names=var_names)
expr
