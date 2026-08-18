function bench_fixture_linear_kf_dynare(workdir, model_file, obs_names, data_file, warmup, reps, output_path)
% Benchmark Dynare's direct Kalman likelihood and smoother for one fixture.

  if nargin ~= 7
    error('Expected workdir, model file, observables, data file, warmup, reps, and output path.');
  end

  previous_dir = pwd;
  restore_dir = onCleanup(@() cd(previous_dir));
  cd(workdir);

  dynare(model_file, 'noclearall', 'nograph');
  global M_ options_ oo_ estim_params_ bayestopt_
  options_.order = 1;
  options_.noprint = 1;
  options_.lik_init = 1;
  options_.kalman_algo = 1;
  [info, oo_, options_, M_] = stoch_simul(M_, options_, oo_, []);
  if info(1) ~= 0
    error('Dynare stochastic solver failed with code %d.', info(1));
  end

  endo_names = cellstr(M_.endo_names);
  n_endo = M_.endo_nbr;
  n_exo = M_.exo_nbr;
  n_obs = numel(obs_names);

  order_var = oo_.dr.order_var(:);
  state_var = oo_.dr.state_var(:);
  ghx = zeros(size(oo_.dr.ghx));
  ghu = zeros(size(oo_.dr.ghu));
  ghx(order_var, :) = oo_.dr.ghx;
  ghu(order_var, :) = oo_.dr.ghu;
  A = zeros(n_endo, n_endo);
  A(:, state_var) = ghx;
  R = ghu;
  Q = M_.Sigma_e;

  obs_idx = zeros(n_obs, 1);
  for i = 1:n_obs
    obs_idx(i) = find(strcmp(endo_names, obs_names{i}), 1);
  end
  Z = zeros(n_obs, n_endo);
  for i = 1:n_obs
    Z(i, obs_idx(i)) = 1.0;
  end
  d = oo_.steady_state(obs_idx);
  if isfield(M_, 'H') && isequal(size(M_.H), [n_obs, n_obs])
    H = M_.H;
  else
    H = zeros(n_obs, n_obs);
  end

  run(data_file);
  first_series = eval(obs_names{1});
  periods = numel(first_series);
  Y = zeros(n_obs, periods);
  for i = 1:n_obs
    series = eval(obs_names{i});
    Y(i, :) = series(:)'-d(i);
  end

  P0 = R * Q * R';
  A_power = A;
  for i = 1:60
    P0 = P0 + A_power * P0 * A_power';
    A_power = A_power * A_power;
  end
  P0 = 0.5 * (P0 + P0');

  run_filter = @() kalman_filter( ...
      Y, 1, periods, zeros(n_endo, 1), P0, ...
      1e-10, 0, false, 0, A, Q, R, H, Z, ...
      n_endo, n_obs, n_exo, 1, 0, {}, false, false);

  for i = 1:warmup
    run_filter();
  end
  likelihood_times = zeros(reps, 1);
  negloglik = NaN;
  for i = 1:reps
    started = tic;
    negloglik = run_filter();
    likelihood_times(i) = toc(started);
  end

  options_.datafile = data_file(1:end-2);
  options_.first_obs = 1;
  options_.nobs = periods;
  options_.prefilter = 0;
  options_.filtered_vars = 1;
  options_.filter_step_ahead = 1;
  var_list_ = {};
  for i = 1:warmup
    [oo_, M_, options_, bayestopt_] = evaluate_smoother( ...
        'calibration', var_list_, M_, oo_, options_, bayestopt_, estim_params_);
  end
  smoother_times = zeros(reps, 1);
  for i = 1:reps
    started = tic;
    [oo_, M_, options_, bayestopt_] = evaluate_smoother( ...
        'calibration', var_list_, M_, oo_, options_, bayestopt_, estim_params_);
    smoother_times(i) = toc(started);
  end

  updated = zeros(numel(oo_.UpdatedVariables.(endo_names{1})), n_endo);
  filtered = zeros(numel(oo_.FilteredVariables.(endo_names{1})), n_endo);
  for i = 1:n_endo
    updated(:, i) = oo_.UpdatedVariables.(endo_names{i})(:);
    filtered(:, i) = oo_.FilteredVariables.(endo_names{i})(:);
  end
  loglik = -negloglik;
  state_names = endo_names;
  declared_endo_nbr = M_.orig_endo_nbr;
  filter_state_nbr = n_endo;
  predetermined_nbr = numel(state_var);
  observable_nbr = n_obs;
  shock_nbr = M_.exo_nbr;
  parameter_nbr = M_.param_nbr;
  save('-v7', output_path, 'likelihood_times', 'smoother_times', 'loglik', ...
       'updated', 'filtered', 'state_names', 'declared_endo_nbr', ...
       'filter_state_nbr', 'predetermined_nbr', 'observable_nbr', ...
       'shock_nbr', 'parameter_nbr');
end
