# type: ignore
# This file is directly copied from linearsolve 3.6.3 source code: https://github.com/lnsongxf/linearsolve/
# Edits have been made to the `klein` function to allow njit compiled matrix operations.

import numpy as np
import scipy.linalg as la
from statsmodels.tools.numdiff import approx_fprime_cs, approx_fprime
from scipy.optimize import root, fsolve, broyden1, broyden2
import warnings
import pandas as pd


from numpy import complex128, float64
from numpy.typing import NDArray
from numba import njit

from typing import TypeAlias, Tuple

NDF: TypeAlias = NDArray[float64]
NDC: TypeAlias = NDArray[complex128]


class model:
    """Defines a class -- linearsolve.model -- with associated methods for solving and simulating dynamic
    stochastic general equilibrium (DSGE) models."""

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
            parameters:         (Pandas Series) Pandas Series object with parameter name strings as the index.
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
            parameters:             (Pandas Series) A Pandas Series with parameter name strings as the
                                        index.
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
        self.parameters = parameters

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

        names["param"] = parameters.index.values

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

    def approximated(self, round=True, precision=4):
        """Returns a string containing the log-linear approximation to the equilibrium conditions

        Args:
            round:      (bool) Whether to round the coefficents in the linear equations. Default: True
            precision:  (int) Number of decimals to round the coefficients. Default: 4

        Returns:
            String with the log-linear approximation to the equilibrium conditions.

        Attributes:
            None

        """

        if round is True:
            a = np.round(self.a, precision)
            b = np.round(self.b, precision)

        leftsides = []
        rightsides = []
        if self.log_linear == True:
            lines = "Log-linear equilibrium conditions:\n\n"
        else:
            lines = "Linear equilibrium conditions:\n\n"

        left_length = 1

        for i in range(self.n_vars):

            left = ""
            right = ""

            left_plus_flag = 0
            right_plus_flag = 0
            if all(np.isclose(0, a[i])):
                left += "0"
            else:
                for j in range(self.n_vars):

                    if not np.isclose(0, a[i][j]):

                        name = self.names["variables"][j]

                        if j > self.n_states - 1:
                            name += "[t+1|t]"

                        else:
                            name += "[t+1]"

                        if np.isclose(1, a[i][j]):
                            coeff = ""

                        elif np.isclose(-1, a[i][j]):
                            coeff = "-"

                        else:
                            coeff = str(a[i][j]) + "·"

                        if left_plus_flag == 0:
                            left += coeff + name
                            left_plus_flag += 1

                        else:
                            if a[i][j] > 0:
                                left += "+" + coeff + name
                            else:
                                left += coeff + name

            if all(np.isclose(0, b[i])):
                right += "0"
            else:
                for j in range(self.n_vars):

                    if not np.isclose(0, b[i][j]):

                        name = self.names["variables"][j] + "[t]"

                        if np.isclose(1, b[i][j]):
                            coeff = ""

                        elif np.isclose(-1, b[i][j]):
                            coeff = "-"

                        else:
                            coeff = str(b[i][j]) + "·"

                        if right_plus_flag == 0:
                            right += coeff + name
                            right_plus_flag += 1

                        else:
                            if b[i][j] > 0:
                                right += "+" + coeff + name
                            else:
                                right += coeff + name

            leftsides.append(left)
            rightsides.append(right)

            if len(left) > left_length:
                left_length = len(left)

        for i in range(self.n_vars):
            leftsides[i] = leftsides[i].rjust(left_length)
            lines += leftsides[i] + " = " + rightsides[i] + "\n\n"

        lines = lines[:-2]

        return lines

    def check_ss(self):
        """Uses Numpy.isclose() to print whether each steady state equilibrium condition evaluates to
        something close to zero.

        Args:
            None

        Returns:
            Numpy ndarray

        Attributes:
            None

        """

        try:
            print(np.isclose(self.equations(self.ss, self.ss, self.parameters), 0))
        except:
            print("Set the steady state first.")

    def compute_ss(self, guess=None, method="fsolve", options={}):
        """Attempts to solve for the steady state of the model.

        Args:
            guess:      (Pandas Series, Numpy array, or list) An initial guess for the
                            steady state solution. The result is highly sensisitve to the intial
                            guess chosen, so be careful. If the guess is a Numpy ndarray or a list
                            then the elements must be ordered to conform with self.names['variables'].
            method:     (str) The function from the Scipy library to use. Your choices are:
                        a. root
                        b. fsolve (default)
                        c. broyden1
                        d. broyden2
            options:    (dict) A dictionary of optional arguments to pass to the numerical solver.
                            Check out the Scipy documentation to see the options available for each routine:
                                http://docs.scipy.org/doc/scipy/reference/optimize.html

        Returns:
            None

        Attributes:
            ss: (Pandas Series) Steady state values of endogenous variables

        """

        if guess is None:
            guess = np.ones(self.n_vars)
        else:
            if isinstance(guess, pd.Series):
                guess = guess.loc[self.names["variables"]]

            elif isinstance(guess, list):
                guess = np.array(guess)

        # Create function for nonlinear solver
        def ss_fun(variables):

            variables = pd.Series(variables, index=self.names["variables"])

            return self.equations(variables, variables, self.parameters)

        def real_ss_fun(variables_transformed):

            z_vals = [
                a + b * 1j
                for a, b in zip(
                    variables_transformed[: int(len(variables_transformed) / 2)],
                    variables_transformed[int(len(variables_transformed) / 2) :],
                )
            ]
            actual_ss_fun = ss_fun(z_vals)
            retval = np.r_[np.real(actual_ss_fun), np.imag(actual_ss_fun)]
            return retval

        if not np.iscomplexobj(self.parameters) and not np.iscomplexobj(guess):

            if method == "fsolve":
                steady_state = fsolve(ss_fun, guess, **options)

            elif method == "root":
                steady_state = root(ss_fun, guess, **options)["x"]

            elif method == "broyden1":
                steady_state = broyden1(ss_fun, guess, **options)

            elif method == "broyden2":
                steady_state = broyden2(ss_fun, guess, **options)

        else:

            if method == "fsolve":
                steady_state = fsolve(
                    real_ss_fun, np.r_[np.real(guess), np.imag(guess)], **options
                )

            elif method == "root":
                steady_state = root(
                    real_ss_fun, np.r_[np.real(guess), np.imag(guess)], **options
                )["x"]

            elif method == "broyden1":
                steady_state = broyden1(
                    real_ss_fun, np.r_[np.real(guess), np.imag(guess)], **options
                )

            elif method == "broyden2":
                steady_state = broyden2(
                    real_ss_fun, np.r_[np.real(guess), np.imag(guess)], **options
                )

            steady_state = [
                a + b * 1j
                for a, b in zip(
                    steady_state[: int(len(steady_state) / 2)],
                    steady_state[int(len(steady_state) / 2) :],
                )
            ]

        # Add ss attribute
        self.ss = pd.Series(steady_state, index=self.names["variables"])

    def impulse(self, T=51, t0=1, shocks=None, center=True, normalize=True):
        """Computes impulse responses for shocks to each state variable.

        Arguments:
                T:          (int) Number of periods to simulate. Default: 51
                t0:         (int) Period in which the shocks are realized. Must be greater than or equal to
                                0. Default: 1
                shocks:     (dict Pandas Series, list or Numpy array) Which shocks to compute impulse responses fror and values
                                If shocks==None shocks
                                is set to a vector of 0.01s. Default = None
                center:     (bool) Subtract steady state for linear approximations (or log steady state for
                                log-linear approximations). Default: True
                normalize:  (bool) Divide simulated data by steady states. Ignored if self.log_linear==True or if
                                        self.ss contains zeros. Default: True if log_linear==False

        Returns
            None

        Attributes:
            irs:    (dict) A dictionary containing Pandas DataFrames. Has the form:
                        self.irs['shock name']['endog var name']

        """
        if normalize and np.any(np.isclose(self.ss, 0)):

            normalize = False
            warnings.warn(
                "Steady state contains zeros so normalize set to False. Set normalize=False to remove this warning.",
                stacklevel=2,
            )

        # Initialize dictionary
        irs_dict = {}

        # Manage shocks
        if isinstance(shocks, pd.Series) or isinstance(shocks, dict):

            if isinstance(shocks, dict):
                shocks = pd.Series(shocks)

            for shock_name in shocks.keys():

                if shock_name not in self.names["shocks"]:
                    warnings.warn(
                        shock_name + " is not in self.names['shocks']", stacklevel=2
                    )

        elif shocks is None:

            shocks = pd.Series(0.01, index=self.names["shocks"])

        else:

            if len(shocks) != self.n_exo_states:
                warnings.warn(
                    "Length of shocks does not equalself.n_exo_states", stacklevel=2
                )

            n_shocks_for_irs = np.min([len(shocks), self.n_exo_states])
            shocks = pd.Series(
                shocks[:n_shocks_for_irs], index=self.names["shocks"][:n_shocks_for_irs]
            )

        for j, shock_name in enumerate(shocks.index):

            s0 = np.zeros([1, self.n_states])
            eps = np.zeros([T, self.n_states])

            eps[t0][j] = shocks[shock_name]

            x = ir(self.f, self.p, eps, s0)

            if center:
                simulated_data = pd.DataFrame(x.T, columns=self.names["variables"])

            else:
                if not self.log_linear:
                    simulated_data = (
                        pd.DataFrame(x.T, columns=self.names["variables"]) + self.ss
                    )

                else:
                    simulated_data = pd.DataFrame(
                        x.T, columns=self.names["variables"]
                    ) + np.log(self.ss)

            if normalize and not self.log_linear:
                simulated_data = simulated_data / self.ss

            simulated_data = pd.concat(
                [pd.Series(eps.T[j], name=shock_name), simulated_data], axis=1
            )

            irs_dict[shock_name] = simulated_data

        self.irs = irs_dict

    def _resolve_steady_state_for_approximation(self, steady_state=None):
        if steady_state is None:

            try:
                steady_state = self.ss
            except:
                raise ValueError(
                    "You must specify a steady state for the model before attempting to linearize."
                )

        return steady_state

    def _linear_approximation_python(self, steady_state):
        def equilibrium(vars_fwd, vars_cur):

            vars_fwd = pd.Series(vars_fwd, index=self.names["variables"])
            vars_cur = pd.Series(vars_cur, index=self.names["variables"])

            equilibrium_left = self.equations(vars_fwd, vars_cur, self.parameters)
            equilibrium_right = np.ones(len(self.names["variables"]))

            return equilibrium_left - equilibrium_right

        equilibrium_fwd = lambda fwd: equilibrium(fwd, steady_state)
        equilibrium_cur = lambda cur: equilibrium(steady_state, cur)

        if not np.iscomplexobj(self.parameters):

            self.a = approx_fprime_cs(steady_state.to_numpy(), equilibrium_fwd)
            self.b = -approx_fprime_cs(steady_state.to_numpy(), equilibrium_cur)

        else:

            self.a = approx_fprime(steady_state.to_numpy(), equilibrium_fwd)
            self.b = -approx_fprime(steady_state.to_numpy(), equilibrium_cur)

    def _log_linear_approximation_python(self, steady_state):
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

        log_equilibrium_fwd = lambda log_fwd: log_equilibrium(log_fwd, np.log(self.ss))
        log_equilibrium_cur = lambda log_cur: log_equilibrium(np.log(self.ss), log_cur)

        if not np.iscomplexobj(self.parameters):

            self.a = approx_fprime_cs(np.log(self.ss).ravel(), log_equilibrium_fwd)
            self.b = -approx_fprime_cs(np.log(self.ss).ravel(), log_equilibrium_cur)

        else:

            self.a = approx_fprime(
                np.log(self.ss).ravel(), log_equilibrium_fwd, centered=True
            )
            self.b = -approx_fprime(
                np.log(self.ss).ravel(), log_equilibrium_cur, centered=True
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

        try:
            self.ss = steady_state[self.names["variables"]]
            self.ss = self.ss.astype(
                np.promote_types(
                    float, np.promote_types(self.parameters.dtype, steady_state.dtype)
                )
            )

        except:
            self.ss = pd.Series(
                steady_state,
                index=self.names["variables"],
                dtype=np.promote_types(
                    float,
                    np.promote_types(
                        self.parameters.dtype, np.array(steady_state).dtype
                    ),
                ),
            )

        self.parameters = self.parameters.astype(
            np.promote_types(self.ss.dtype, self.parameters.dtype)
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

    def solved(self, round=True, precision=4):
        """Returns a string containing the solution to the linear system

        Args:
            round:       (bool) Whether to round the coefficents in the solution equations. Default: True
            precisions:  (int) Number of decimals to round the coefficients. Default: 4

        Returns:
            String with the linear approximation to the equilibrium conditions.

        Attributes:
            None

        """

        if round is True:
            f = np.round(self.f, precision)
            p = np.round(self.p, precision)

        leftsides = []
        rightsides = []
        if self.log_linear == True:
            lines = "Solution to the log-linear system:\n\n"
        else:
            lines = "Solution to the linear system:\n\n"

        left_length = 1

        for i in range(self.n_states):

            left = ""
            right = ""
            right_plus_flag = 0

            left += self.names["variables"][i] + "[t+1]"

            if all(np.isclose(0, p[i])):
                right += self.names["shocks"][i] + "[t+1]"
            else:
                for j in range(self.n_states):

                    if not np.isclose(0, p[i][j]):

                        if right_plus_flag == 0:
                            right += (
                                str(p[i][j]) + "·" + self.names["variables"][j] + "[t]"
                            )
                            right_plus_flag += 1

                        else:
                            if p[i][j] > 0:
                                right += (
                                    "+"
                                    + str(p[i][j])
                                    + "·"
                                    + self.names["variables"][j]
                                    + "[t]"
                                )

                            else:
                                right += (
                                    str(p[i][j])
                                    + "·"
                                    + self.names["variables"][j]
                                    + "[t]"
                                )
                if i < self.n_exo_states:
                    right += "+" + self.names["shocks"][i] + "[t+1]"
            leftsides.append(left)
            rightsides.append(right)

            if len(left) > left_length:
                left_length = len(left)

        for i in range(self.n_vars - self.n_states):

            left = ""
            right = ""
            right_plus_flag = 0

            left += self.names["variables"][self.n_states + i] + "[t]"

            if all(np.isclose(0, f[i])):
                right += "0"
            else:
                for j in range(self.n_states):
                    if not np.isclose(0, f[i][j]):

                        name = self.names["variables"][j] + "[t]"

                        if np.isclose(1, f[i][j]):
                            coeff = ""

                        elif np.isclose(-1, f[i][j]):
                            coeff = "-"

                        else:
                            coeff = str(f[i][j]) + "·"

                        if right_plus_flag == 0:
                            right += coeff + name
                            right_plus_flag += 1

                        else:
                            if f[i][j] > 0:
                                right += "+" + coeff + name
                            else:
                                right += coeff + name

            leftsides.append(left)
            rightsides.append(right)

            if len(left) > left_length:
                left_length = len(left)

        for i in range(self.n_vars):
            leftsides[i] = leftsides[i].rjust(left_length)
            lines += leftsides[i] + " = " + rightsides[i] + "\n\n"
        lines = lines[:-2]

        return lines

    def stoch_sim(
        self,
        T=51,
        drop_first=300,
        covariance_matrix=None,
        variances=None,
        seed=None,
        center=True,
        normalize=True,
    ):
        """Computes a stohcastic simulation of the model.

        Arguments:
                T:                  (int) Number of periods to simulate. Default: 51
                drop_first:         (int) Number of periods to simulate before generating the simulated periods.
                                Default: 300
                covariance_matrix:  (list or Numpy.ndarray) Covariance matrix shocks. If there is only one shock,
                                        either a number or 1-d array may be supplied. If not given, exogenous
                                        shock standard deviations are set to 0.01.
                variances           (list or Numpy.ndarray) The variances of the exogenous shocks. Will be used to
                                        form the covariance matrix. If variances and covariance_matrix are both
                                        supplied, the latter will take precendence.
                seed:               (int) Sets the seed for the Numpy random number generator. Default: None
                center:             (bool) Subtract steady state for linear approximations (or log steady state for
                                        log-linear approximations). Default: True
                normalize:  (bool) Divide simulated data by steady states. Ignored if self.log_linear==True or if
                                        self.ss contains zeros. Default: True if log_linear==False

        Returns
            None

        Attributes:
            simulated:    (Pandas DataFrame)

        """

        if normalize and np.any(np.isclose(self.ss, 0)):

            normalize = False
            warnings.warn(
                "Steady state contains zeros so normalize set to False. Set normalize=False to remove this warning.",
                stacklevel=2,
            )

        # Initialize states
        s0 = np.zeros([1, self.n_states])

        # Set covariance_matrix if not given
        if covariance_matrix is not None:
            covariance_matrix = np.atleast_2d(covariance_matrix)

        elif variances is not None:
            covariance_matrix = np.diag(np.r_[variances])

        else:
            covariance_matrix = np.diag(self.n_exo_states * [0.01**2])

        if len(covariance_matrix) != self.n_exo_states:
            raise Exception(
                "Length of covariance_matrix doesn't match number of exogenous states"
            )

        # Set seed for the Numpy random number generator
        if seed is None:
            rng = np.random.default_rng()

        else:
            rng = np.random.default_rng(seed=seed)

        # Simulate shocks
        eps = np.zeros([drop_first + T, self.n_states])
        eps[:, : len(covariance_matrix)] = rng.multivariate_normal(
            mean=np.zeros(len(covariance_matrix)),
            cov=covariance_matrix,
            size=[drop_first + T],
        )

        # Compute responses given shocks
        x = ir(self.f, self.p, eps, s0)

        # Construct DataFrame
        if center:
            simulated_data = pd.DataFrame(
                x.T[drop_first:], columns=self.names["variables"]
            )

        else:
            if not self.log_linear:
                simulated_data = (
                    pd.DataFrame(x.T[drop_first:], columns=self.names["variables"])
                    + self.ss
                )

            else:
                simulated_data = pd.DataFrame(
                    x.T[drop_first:], columns=self.names["variables"]
                ) + np.log(self.ss)

        if normalize and not self.log_linear:
            simulated_data = simulated_data / self.ss

        simulated_data = pd.concat(
            [
                pd.DataFrame(
                    eps[drop_first:, : self.n_exo_states], columns=self.names["shocks"]
                ),
                simulated_data,
            ],
            axis=1,
        )

        self.simulated = simulated_data


### End of model class ####################################################################################


def ir(f, p, eps, s0=None):
    """Simulates a model in the following form:

            u(t)   = f*s(t) + e(t)
            s(t+1) = p*s(t)

    where s(t) is an (n_states x 1) vector of state variables, u(t) is an (n_costates x 1) vector of costate
    variables, and e(t) is an (n_states x 1) vector of exogenous shocks.

    Args:
        f:      (Numpy ndarray) Coefficnent matrix of appropriate size
        p:      (Numpy ndarray) Coefficnent matrix of appropriate size
        eps:    (Numpy ndarray) T x n_states array of exogenous shocks.
        s0:     (Numpy ndarray) 1 x n_states array of zeros of initial state value. Optional; Default: 0.

    Returns
        s:   (Numpy ndarray) states simulated from t = 0,1,...,T-1
        u:   (Numpy ndarray) costates simulated from t = 0,1,...,T-1

    """

    T = np.max(eps.shape)
    n_states = np.shape(p)[0]
    n_costates = np.shape(f)[0]

    if s0 is None:

        s0 = np.zeros([1, n_states])

    s = np.zeros([T + 1, n_states], dtype=np.promote_types(f.dtype, p.dtype))
    u = np.zeros([T, n_costates], dtype=np.promote_types(f.dtype, p.dtype))

    s[0] = s0

    for i, e in enumerate(eps):
        s[i + 1] = p.dot(s[i]) + e
        u[i] = f.dot(s[i + 1])

    s = s[1:]

    return np.concatenate((s.T, u.T))


def _as_1d_array(values):
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()

    return np.asarray(values).reshape(-1)


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
