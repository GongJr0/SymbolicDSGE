# type: ignore
# This file is directly copied from linearsolve 3.6.3 source code: https://github.com/lnsongxf/linearsolve/
# Edits have been made to the `klein` function to allow njit compiled matrix operations.

import numpy as np
import scipy.linalg as la
from statsmodels.tools.numdiff import approx_fprime_cs, approx_fprime


from numpy import complex128, float64
from numpy.typing import NDArray
from numba import njit

from collections.abc import Mapping
from typing import Any, TypeAlias, Tuple

NDF: TypeAlias = NDArray[float64]
NDC: TypeAlias = NDArray[complex128]

_PANDAS = None


def _require_pandas():
    global _PANDAS
    if _PANDAS is None:
        import pandas as _pd

        _PANDAS = _pd

    return _PANDAS


def _stringify_names(names):
    return [
        name if isinstance(name, str) else getattr(name, "name", str(name))
        for name in names
    ]


def _is_labeled_vector(values):
    index = getattr(values, "index", None)
    return index is not None and not callable(index) and not hasattr(values, "columns")


def _value_dtype(values):
    if isinstance(values, Mapping):
        if len(values) == 0:
            return np.dtype(float64)
        return np.asarray(list(values.values())).dtype

    if hasattr(values, "to_numpy"):
        return np.asarray(values.to_numpy()).dtype

    return np.asarray(values).dtype


def _normalize_parameter_input(parameters, parameter_names=None):
    if parameters is None:
        names = [] if parameter_names is None else list(parameter_names)
        return np.empty((0,), dtype=float64), names

    if isinstance(parameters, Mapping):
        inferred_names = _stringify_names(parameters.keys())
        names = inferred_names if parameter_names is None else list(parameter_names)
        if len(names) != len(inferred_names):
            raise ValueError("parameter_names length must match parameters length.")
        return parameters, names

    if _is_labeled_vector(parameters):
        inferred_names = _stringify_names(getattr(parameters, "index"))
        names = inferred_names if parameter_names is None else list(parameter_names)
        if len(names) != len(inferred_names):
            raise ValueError("parameter_names length must match parameters length.")
        return parameters, names

    array = np.ascontiguousarray(np.asarray(parameters).reshape(-1))
    if parameter_names is None:
        raise ValueError("parameter_names is required when parameters are array-like.")

    names = list(parameter_names)
    if array.shape[0] != len(names):
        raise ValueError("parameter_names length must match parameters length.")

    return array, names


def _normalize_named_vector(values, names, dtype=None):
    if isinstance(values, Mapping):
        try:
            ordered = [values[name] for name in names]
        except KeyError as exc:
            raise KeyError(f"Missing named value '{exc.args[0]}'.") from exc
        array = np.asarray(ordered, dtype=dtype)
    elif _is_labeled_vector(values):
        if hasattr(values, "loc"):
            selected = values.loc[list(names)]
        else:
            selected = values[list(names)]

        if hasattr(selected, "to_numpy"):
            array = selected.to_numpy(dtype=dtype)
        else:
            array = np.asarray(selected, dtype=dtype)
    else:
        array = np.asarray(values, dtype=dtype)

    vector = np.ascontiguousarray(array).reshape(-1)
    if vector.shape[0] != len(names):
        raise ValueError(
            f"Expected vector of length {len(names)}, received {vector.shape[0]}."
        )

    return vector


def _cast_parameter_values(parameters, dtype):
    if isinstance(parameters, Mapping):
        return {
            key: np.asarray(value, dtype=dtype).reshape(()).item()
            for key, value in parameters.items()
        }

    if hasattr(parameters, "astype"):
        casted = parameters.astype(dtype)
        if isinstance(casted, np.ndarray):
            return np.ascontiguousarray(casted.reshape(-1))
        return casted

    return np.ascontiguousarray(np.asarray(parameters, dtype=dtype).reshape(-1))


class model:
    """Defines a minimal linearsolve-compatible model used for approximation and Klein solving."""

    def __init__(
        self,
        equations=None,
        variables=None,
        costates=None,
        states=None,
        exo_states=None,
        endo_states=None,
        shock_names=None,
        parameters=None,
        parameter_names=None,
        shock_prefix=None,
        n_states=None,
        n_exo_states=None,
    ):
        """Initializing an instance linearsolve.model requires values for the following variables:

        Args:
            equations:          (fun) A function that represents the equilibirum conditions for a DSGE model.
                                    The function should accept three arguments:
                                        * vars_fwd:     endogenous variables dated t+1
                                        * vars_cur:     endogenous variables dated t
                                        * parameters:   the parameters of the model
                                    The function should return an n-dimensional array with each element of
                                    the returned array being equaling an equilibrium condition of the model
                                    solved for zero.
            variables:          (list) A list of strings with the names of the endogenous variables. The
                                    state variables with exogenous shocks must be ordered first, followed by state
                                    variables without exogenous shocks, followed by control variables. E.g., for a
                                    3-variables RBC model, var_names = ['a','k','c']
            costates       (list or str) A list of the costate variables of the model. May be a string if only
                                    one costate variable.
            states:         (list or str) A list of the state variables of the model. May be a string if only
                                    one state variable.
            exo_states:     (list or str) A list of the state variables of the models that have exogenous shocks.
                                    May be a string if only one exogenous state variable.
            endo_states:     (list or str) A list of the state variables of the models that do not have exogenous shocks.
                                    May be a string if only one endogenous state variable.
            shock_names:        (list) A list of strings with the names of the exogenous shocks to each state
                                    variable. The order of names must agree with the relevant elements of var_names.
            n_states:           (int) The number of state variables in the model.
            n_exo_states:       (int) The number of state variables with exogenous shocks. If None, then it's assumed
                                    that all state variables have exogenous shocks. Default: None
            parameters:         Parameter values. May be a mapping, labeled vector, or array-like.
            parameter_names:    (list) Parameter names corresponding to array-like `parameters`.
            shock_prefix:       (str) By default shocks are named 'e_[var1]', 'e_[var2]', etc. Change the prefix 'e_'
                                    with this parameter or set the shock_names parameter to avoid prefixes altogether.

        Returns:
            None

        Attributes:
            equations:   (fun) Function that returns the equilibrium comditions of the model.
            n_vars:                 (int) The number of variables in the model.
            n_states:               (int) The number of state variables in the model.
            n_exo_states:           (int) The number of exogenous state variables.
            n_endo_states:          (int) The number of endogenous state variables.
            n_costates:             (int) The number of costate or control variables in the model.
            names:                  (dict) A dictionary with keys 'variables', 'shocks', and 'param' that
                                        stores the names of the model's variables, shocks, and parameters.
            parameters:             Parameter values supplied at construction.
        """

        if variables is None:
            if costates is None:
                costates = []
                self.n_costates = 0

            else:
                costates = np.array(costates).flatten()
                self.n_costates = len(costates)

            if states is None:
                if exo_states is None:
                    exo_states = []
                    self.n_exo_states = 0

                else:
                    exo_states = np.array(exo_states).flatten()
                    self.n_exo_states = len(exo_states)

                if endo_states is None:
                    endo_states = []
                    self.n_endo_states = 0

                else:
                    endo_states = np.array(endo_states).flatten()
                    self.n_endo_states = len(endo_states)

                states = np.r_[exo_states, endo_states]
                self.n_states = self.n_endo_states + self.n_exo_states

            else:
                states = np.array(states).flatten()
                self.n_states = len(states)

                if exo_states is not None:
                    exo_states = np.array(exo_states).flatten()

                    self.n_exo_states = len(exo_states)
                    self.n_endo_states = self.n_states - self.n_exo_states
                    states = np.r_[
                        states[np.isin(states, exo_states)],
                        states[~np.isin(states, exo_states)],
                    ]

                elif endo_states is not None:
                    endo_states = np.array(endo_states).flatten()

                    self.n_endo_states = len(endo_states)
                    self.n_exo_states = self.n_states - self.n_endo_states
                    states = np.r_[
                        states[~np.isin(states, endo_states)],
                        states[np.isin(states, endo_states)],
                    ]

                else:
                    self.n_exo_states = self.n_states
                    self.n_endo_states = self.n_states - self.n_exo_states

            self.n_vars = self.n_costates + self.n_states

        else:
            variables = np.array(variables).flatten()
            self.n_vars = len(variables)

            if states is None and exo_states is None and endo_states is None:

                if n_states is None:

                    if n_exo_states is not None:
                        self.n_exo_states = n_exo_states

                    else:
                        self.n_exo_states = 0

                    self.n_states = self.n_exo_states
                    self.n_endo_states = self.n_states - self.n_exo_states
                    self.n_costates = self.n_vars - self.n_states

                    states = variables[:n_exo_states]
                    exo_states = variables[n_exo_states:n_states]
                    costates = variables[n_states:]

                else:

                    self.n_states = n_states

                    if n_exo_states is not None:
                        self.n_exo_states = n_exo_states

                    else:
                        self.n_exo_states = 0

                self.n_endo_states = self.n_states - self.n_exo_states
                self.n_costates = self.n_vars - self.n_states

                exo_states = variables[:n_exo_states]
                endo_states = variables[n_exo_states:n_states]
                states = variables[:n_states]
                costates = variables[n_states:]

            else:
                if states is None:
                    if exo_states is None:
                        exo_states = []
                        self.n_exo_states = 0

                    else:
                        exo_states = np.array(exo_states).flatten()
                        self.n_exo_states = len(exo_states)

                    if endo_states is None:
                        endo_states = []
                        self.n_endo_states = 0

                    else:
                        endo_states = np.array(endo_states).flatten()
                        self.n_endo_states = len(endo_states)

                    states = np.r_[exo_states, endo_states]
                    self.n_states = self.n_endo_states + self.n_exo_states

                else:
                    states = np.array(states).flatten()
                    self.n_states = len(states)

                    if exo_states is not None:
                        exo_states = np.array(exo_states).flatten()
                        self.n_exo_states = len(exo_states)
                        self.n_endo_states = self.n_states - self.n_exo_states
                        states = np.r_[
                            states[np.isin(states, exo_states)],
                            states[~np.isin(states, exo_states)],
                        ]

                    elif endo_states is not None:
                        endo_states = np.array(endo_states).flatten()
                        self.n_endo_states = len(endo_states)
                        self.n_exo_states = self.n_states - self.n_endo_states
                        states = np.r_[
                            states[~np.isin(states, endo_states)],
                            states[np.isin(states, endo_states)],
                        ]

                    else:
                        self.n_exo_states = self.n_states
                        self.n_endo_states = self.n_states - self.n_exo_states
                        exo_states = states
                        endo_states = np.array([])

                self.n_costates = self.n_vars - self.n_states

                costates = variables[~np.isin(variables, states)]

        self.equations = equations
        self.parameters, parameter_names = _normalize_parameter_input(
            parameters,
            parameter_names=parameter_names,
        )

        names = {}

        names["variables"] = np.r_[states, costates].tolist()

        if shock_names is not None:

            names["shocks"] = shock_names

        else:
            if shock_prefix is None:
                shock_prefix = "e_"

            names["shocks"] = np.array(
                [shock_prefix + names["variables"][i] for i in range(self.n_exo_states)]
            )

        if len(names["shocks"]) != self.n_exo_states:
            raise Exception(
                "Length of shock_names doesn't match number of exogenous states"
            )

        names["param"] = list(parameter_names)

        self.names = names

    # Methods

    def approximate_and_solve(self, log_linear=False, eigenvalue_warnings=True):
        """Method approximates and solves a dynamic stochastic general equilibrium (DSGE) model by
        constructing the linear or log-linear approximation and solving the model
        using Klein's (2000) method.

        Args:
            log_linear:             (bool) Whether to compute log-linear or linear approximation. Default: False
            eigenvalue_warnings:    (bool) Whether to print warnings that there are too many or few eigenvalues. Default: True

        Returns:
            None

        Attributes:
            a:          (Numpy ndarray) Coefficient matrix on forward-dated variables.
            b:          (Numpy ndarray) Coefficient matrix on current-dated variables.
            f:          (Numpy ndarray) Solution matrix coeffients on s(t) in control equation.
            p:          (Numpy ndarray) Solution matrix coeffients on s(t) in state equation.
            stab:       (int) Indicates solution stability and uniqueness
                            stab == 1: too many stable eigenvalues
                            stab == -1: too few stable eigenvalues
                            stab == 0: just enoughstable eigenvalues
            eig:        The generalized eigenvalues from the Schur decomposition
            log_linear: (bool) Whether the model is log-linear. Sets to log-linear.

        """

        # Set attribute
        self.log_linear = log_linear

        # Approximate
        if log_linear == True:
            self.log_linear_approximation()
        else:
            self.linear_approximation()

        # Solve the model
        self.solve_klein(self.a, self.b, eigenvalue_warnings=eigenvalue_warnings)

    def _resolve_steady_state_for_approximation(self, steady_state=None):
        if steady_state is None:

            try:
                steady_state = self.ss
            except:
                raise ValueError(
                    "You must specify a steady state for the model before attempting to linearize."
                )

        return _normalize_named_vector(steady_state, self.names["variables"])

    def _linear_approximation_python(self, steady_state):
        steady_state_array = _as_1d_array(steady_state)
        pd = _require_pandas()

        def equilibrium(vars_fwd, vars_cur):

            vars_fwd = pd.Series(vars_fwd, index=self.names["variables"])
            vars_cur = pd.Series(vars_cur, index=self.names["variables"])

            equilibrium_left = self.equations(vars_fwd, vars_cur, self.parameters)
            equilibrium_right = np.ones(len(self.names["variables"]))

            return equilibrium_left - equilibrium_right

        equilibrium_fwd = lambda fwd: equilibrium(fwd, steady_state)
        equilibrium_cur = lambda cur: equilibrium(steady_state, cur)

        if not np.iscomplexobj(self.parameters):

            self.a = approx_fprime_cs(steady_state_array, equilibrium_fwd)
            self.b = -approx_fprime_cs(steady_state_array, equilibrium_cur)

        else:

            self.a = approx_fprime(steady_state_array, equilibrium_fwd)
            self.b = -approx_fprime(steady_state_array, equilibrium_cur)

    def _log_linear_approximation_python(self, steady_state):
        steady_state_array = _as_1d_array(steady_state)
        log_steady_state = np.log(steady_state_array)
        pd = _require_pandas()

        def log_equilibrium(log_vars_fwd, log_vars_cur):

            log_vars_fwd = pd.Series(log_vars_fwd, index=self.names["variables"])
            log_vars_cur = pd.Series(log_vars_cur, index=self.names["variables"])

            equilibrium_left = (
                self.equations(
                    np.exp(log_vars_fwd), np.exp(log_vars_cur), self.parameters
                )
                + 1
            )
            equilibrium_right = np.ones(len(self.names["variables"]))

            return np.log(equilibrium_left) - np.log(equilibrium_right)

        log_equilibrium_fwd = lambda log_fwd: log_equilibrium(log_fwd, log_steady_state)
        log_equilibrium_cur = lambda log_cur: log_equilibrium(log_steady_state, log_cur)

        if not np.iscomplexobj(self.parameters):

            self.a = approx_fprime_cs(log_steady_state.ravel(), log_equilibrium_fwd)
            self.b = -approx_fprime_cs(log_steady_state.ravel(), log_equilibrium_cur)

        else:

            self.a = approx_fprime(
                log_steady_state.ravel(), log_equilibrium_fwd, centered=True
            )
            self.b = -approx_fprime(
                log_steady_state.ravel(), log_equilibrium_cur, centered=True
            )

    def _numeric_approximation(self, steady_state, log_linear):
        equations_numeric = getattr(self, "_equations_numeric", None)
        parameter_array = getattr(self, "_parameter_array", None)
        if equations_numeric is None or parameter_array is None:
            return None

        steady_state_array = _as_1d_array(steady_state)
        parameter_array = _as_1d_array(parameter_array)

        if np.iscomplexobj(steady_state_array) and not np.all(
            np.isclose(np.imag(steady_state_array), 0.0)
        ):
            return None

        if np.iscomplexobj(parameter_array) and not np.all(
            np.isclose(np.imag(parameter_array), 0.0)
        ):
            return None

        steady_state_real = np.ascontiguousarray(
            np.real(steady_state_array).astype(float64)
        )
        parameter_real = np.ascontiguousarray(np.real(parameter_array).astype(float64))

        if log_linear and np.any(steady_state_real <= 0.0):
            return None

        return _approximate_system_numeric(
            equations_numeric,
            steady_state_real,
            parameter_real,
            log_linear,
        )

    def linear_approximation(self, steady_state=None):
        """Given a nonlinear rational expectations model in the form:

                    psi_1[x(t+1),x(t)] = psi_2[x(t+1),x(t)]

            this method returns the linear approximation of the model with matrices a and b such that:

                    a * y(t+1) = b * y(t)

            where y(t) = x(t) - x is the log deviation of the vector x from its steady state value.

        Args:
            steady_state:   (Pandas Series or numpy array or list)

        Returns:
            None

        Attributes:
            log_linear:     (bool) Whether the model is log-linear. Sets to False.
            a:              (Numpy ndarray)
            b:              (Numpy ndarray)

        """

        # Set log_linear attribute
        self.log_linear = False

        steady_state = self._resolve_steady_state_for_approximation(steady_state)
        approx = self._numeric_approximation(steady_state, log_linear=False)
        if approx is not None:
            self.a, self.b = approx
            return

        self._linear_approximation_python(steady_state)

    def log_linear_approximation(self, steady_state=None):
        """Given a nonlinear rational expectations model in the form:

                    psi_1[x(t+1),x(t)] = psi_2[x(t+1),x(t)]

            this method returns the log-linear approximation of the model with matrices a and b such that:

                    a * y(t+1) = b * y(t)

            where y(t) = log x(t) - log x is the log deviation of the vector x from its steady state value.

        Args:
            steady_state:   (Pandas Series or numpy array)

        Returns:
            None

        Attributes:
            log_linear:     (bool) Whether the model is log_linear. Sets to True.
            a:              (Numpy ndarray)
            b:              (Numpy ndarray)

        """

        # Set log_linear attribute
        self.log_linear = True

        steady_state = self._resolve_steady_state_for_approximation(steady_state)
        approx = self._numeric_approximation(steady_state, log_linear=True)
        if approx is not None:
            self.a, self.b = approx
            return

        self._log_linear_approximation_python(steady_state)

    def set_ss(self, steady_state):
        """Directly set the steady state of the model.

        Args:
            steady_state:   (Pandas Series, Numpy array, or list)

        Returns:
            None

        Attributes:
            ss: (Pandas Series) Steady state values of endogenous variables

        """

        parameter_dtype = _value_dtype(self.parameters)
        steady_state_dtype = _value_dtype(steady_state)
        promoted_dtype = np.promote_types(
            float,
            np.promote_types(parameter_dtype, steady_state_dtype),
        )
        self.ss = _normalize_named_vector(
            steady_state,
            self.names["variables"],
            dtype=promoted_dtype,
        )

        self.parameters = _cast_parameter_values(
            self.parameters,
            np.promote_types(self.ss.dtype, parameter_dtype),
        )

    def solve_klein(self, a=None, b=None, eigenvalue_warnings=True):
        """Solves a linear rational expectations model of the form:

                a * x(t+1) = b * x(t) + e(t)

        The method returns the solution to the law of motion:

                u(t)   = f*s(t) + e(t)
                s(t+1) = p*s(t)

        Args:
            a:                      (Numpy ndarray) coefficient matrix
            b:                      (Numpy ndarray) coefficient matrix
            eigenvalue_warnings:    (bool) Whether to print warnings that there are too many or few eigenvalues. Default: True

        Returns:
            None

        Attributes:
            f:      (Numpy ndarray) Solution matrix coeffients on s(t)
            p:      (Numpy ndarray) Solution matrix coeffients on s(t)
            stab:   (int) Indicates solution stability and uniqueness
                        stab == 1: too many stable eigenvalues
                        stab == -1: too few stable eigenvalues
                        stab == 0: just enough stable eigenvalues
            eig:    The generalized eigenvalues from the Schur decomposition

        """

        if a is None and b is None:

            a = self.a
            b = self.b

        self.f, n, self.p, l, self.stab, self.eig = klein(
            a=a,
            b=b,
            c=None,
            _phi=None,
            n_states=self.n_states,
        )

        if not np.iscomplexobj(self.parameters):

            self.f = np.real(self.f)
            self.p = np.real(self.p)
            l = np.real(l)
            n = np.real(n)

            # self.f = np.abs(self.f)
            # self.p = np.abs(self.p)
            # l = np.abs(l)
            # n = np.abs(n)


### End of model class ####################################################################################


def _as_1d_array(values):
    if isinstance(values, Mapping):
        values = list(values.values())
    elif hasattr(values, "to_numpy"):
        values = values.to_numpy()

    return np.ascontiguousarray(np.asarray(values).reshape(-1))


@njit
def _evaluate_equilibrium_numeric(eq_func, fwd, cur, params, log_linear):
    if log_linear:
        return np.log(eq_func(np.exp(fwd), np.exp(cur), params) + 1.0)

    return eq_func(fwd, cur, params)


@njit
def _complex_step_jacobian(eq_func, base_point, params, log_linear, differentiate_fwd):
    step = float64(1e-30)
    complex_step = complex128(1j * step)
    base_complex = np.ascontiguousarray(base_point.astype(complex128))
    params_complex = np.ascontiguousarray(params.astype(complex128))
    base_residual = _evaluate_equilibrium_numeric(
        eq_func,
        base_complex,
        base_complex,
        params_complex,
        log_linear,
    )
    jac = np.empty((base_residual.shape[0], base_point.shape[0]), dtype=float64)

    for j in range(base_point.shape[0]):
        fwd = base_complex.copy()
        cur = base_complex.copy()

        if differentiate_fwd:
            fwd[j] = fwd[j] + complex_step
        else:
            cur[j] = cur[j] + complex_step

        residual = _evaluate_equilibrium_numeric(
            eq_func,
            fwd,
            cur,
            params_complex,
            log_linear,
        )
        jac[:, j] = np.imag(residual) / step

    return jac


@njit
def _approximate_system_numeric(eq_func, steady_state, params, log_linear):
    base_point = np.ascontiguousarray(steady_state.astype(float64))
    parameter_vector = np.ascontiguousarray(params.astype(float64))

    if log_linear:
        base_point = np.ascontiguousarray(np.log(base_point))

    a = _complex_step_jacobian(eq_func, base_point, parameter_vector, log_linear, True)
    b = -_complex_step_jacobian(
        eq_func,
        base_point,
        parameter_vector,
        log_linear,
        False,
    )

    return a, b


@njit(cache=True)
def _to_complex(
    s: NDC | NDF,
    t: NDC | NDF,
    alpha: NDC | NDF,
    beta: NDC | NDF,
    q: NDC | NDF,
    z: NDC | NDF,
) -> Tuple[NDC, NDC, NDC, NDC, NDC, NDC]:
    return (
        s.astype(complex128),
        t.astype(complex128),
        alpha.astype(complex128),
        beta.astype(complex128),
        q.astype(complex128),
        z.astype(complex128),
    )


@njit(cache=True)
def _klein_postprocess(
    s: NDC,
    t: NDC,
    q: NDC,
    z: NDC,
    c: NDC | NDF,
    _phi: NDC | NDF,
    n_states: int,
) -> Tuple[
    NDC,
    NDC,
    NDC,
    NDC,
    int,
    NDC,
]:
    forcingVars = False

    nz: int = 0
    if c.size == 0:
        phi = np.empty((0, 0), dtype=complex128)
        c_complex = np.empty((0, 0), dtype=complex128)
    else:
        phi = np.ascontiguousarray(_phi.astype(complex128))
        c_complex = np.ascontiguousarray(c.astype(complex128))
        forcingVars = True
        nz = c.shape[1]

    z11: NDC = np.ascontiguousarray(z[:n_states, :n_states])
    z12: NDC = np.ascontiguousarray(z[:n_states, n_states:])
    z21: NDC = np.ascontiguousarray(z[n_states:, :n_states])
    z22: NDC = np.ascontiguousarray(z[n_states:, n_states:])

    n_costates = s.shape[0] - n_states

    s11: NDC = np.ascontiguousarray(s[:n_states, :n_states])
    if n_states > 0:
        z11i: NDC = np.ascontiguousarray(np.linalg.inv(z11))
        s11i: NDC = np.ascontiguousarray(np.linalg.inv(s11))
    else:
        z11i = z11
        s11i = s11

    s12: NDC = np.ascontiguousarray(s[:n_states, n_states:])
    s22: NDC = np.ascontiguousarray(s[n_states:, n_states:])
    t11: NDC = np.ascontiguousarray(t[:n_states, :n_states])
    t12: NDC = np.ascontiguousarray(t[:n_states, n_states:])
    t22: NDC = np.ascontiguousarray(t[n_states:, n_states:])
    q1: NDC = np.ascontiguousarray(q[:n_states, :])
    q2: NDC = np.ascontiguousarray(q[n_states:, :])

    stab: int = 0

    if n_states > 0:
        if np.abs(t[n_states - 1, n_states - 1]) > np.abs(
            s[n_states - 1, n_states - 1]
        ):
            stab = -1  # Too few stable

    if n_states < n_states + n_costates:
        if np.abs(t[n_states, n_states]) < np.abs(s[n_states, n_states]):
            stab = 1  # Too many unstable

    tii: NDC = np.diag(t)
    sii: NDC = np.diag(s)
    eig: NDC = np.zeros(tii.shape, dtype=complex128)

    for k in range(tii.shape[0]):
        if np.abs(sii[k]) > 1e-12:
            eig[k] = tii[k] / sii[k]
        else:
            eig[k] = complex128(np.inf)

    if n_states > 0:
        dyn = np.ascontiguousarray(np.linalg.solve(s11, t11))
    else:
        dyn = np.empty((0, 0), dtype=complex128)

    f = z21.dot(z11i)
    p = z11.dot(dyn).dot(z11i)

    if not forcingVars:
        n = np.empty((n_costates, 0), dtype=complex128)
        l = np.empty((n_states, 0), dtype=complex128)
    else:
        phiT = np.ascontiguousarray(phi.T)
        nzI = np.ascontiguousarray(np.identity(nz, dtype=complex128))
        mat1: NDC = np.kron(phiT, s22) - np.kron(nzI, t22)
        mat1i: NDC = np.linalg.inv(mat1)
        q2c = q2.dot(c_complex)
        vecq2c = np.ascontiguousarray(q2c.flatten().T)
        vecm: NDC = mat1i.dot(vecq2c)
        m: NDC = np.ascontiguousarray(vecm.T.reshape(nz, n_costates))
        n: NDC = (z22 - z21.dot(z11i).dot(z12)).dot(m)
        l: NDC = (
            -z11.dot(s11i).dot(t11).dot(z12).dot(m)
            + z11.dot(s11i).dot(t12.dot(m) - s12.dot(m).dot(phi) + q1.dot(c_complex))
            + z12.dot(m).dot(phi)
        )

    return f, n, p, l, stab, eig


def _normalize_optional_matrix(mat):
    if mat is None:
        return np.empty((0, 0), dtype=float64)

    arr = np.asarray(mat)
    if arr.size == 0:
        return np.empty((0, 0), dtype=arr.dtype)

    if arr.ndim == 1:
        arr = arr.reshape(arr.shape[0], 1)

    return np.ascontiguousarray(arr)


def klein(a: NDF, b: NDF, c: NDF | None, _phi: NDF | None, n_states: int) -> Tuple[
    NDC,
    NDC,
    NDC,
    NDC,
    int,
    NDC,
]:
    has_c = c is not None and np.asarray(c).size > 0
    has_phi = _phi is not None and np.asarray(_phi).size > 0
    if has_c != has_phi:
        raise ValueError("c and _phi must either both be provided or both be empty.")

    c_arr = _normalize_optional_matrix(c)
    phi_arr = _normalize_optional_matrix(_phi)

    _s, _t, _alpha, _beta, _q, _z = la.ordqz(a, b, sort="ouc", output="complex")
    s, t, _alpha, _beta, q, z = _to_complex(
        _s,
        _t,
        _alpha,
        _beta,
        _q,
        _z,
    )
    return _klein_postprocess(s, t, q, z, c_arr, phi_arr, n_states)
