function bench_sw2007_linear_kf_dynare(workdir, warmup, reps, output_path)
% Benchmark Dynare's SW2007 likelihood and smoother entries.

  if nargin ~= 4
    error('Expected workdir, warmup, reps, and output_path.');
  end

  previous_dir = pwd;
  restore_dir = onCleanup(@() cd(previous_dir));
  cd(workdir);

  dynare sw2007.mod noclearall nograph;
  global M_ options_ oo_ estim_params_ bayestopt_
  options_.order = 1;
  options_.noprint = 1;
  [info, oo_, options_, M_] = stoch_simul(M_, options_, oo_, []);
  if info(1) ~= 0
    error('Dynare stochastic solver failed with code %d.', info(1));
  end

  endo_names = cellstr(M_.endo_names);
  obs_names = {'dy', 'dc', 'dinve', 'labobs', 'pinfobs', 'dw', 'robs'};
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

  run('sw2007_kf_data.m');
  Y = [dy, dc, dinve, labobs, pinfobs, dw, robs]' - d * ones(1, numel(dy));
  periods = size(Y, 2);

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

  options_.datafile = 'sw2007_kf_data';
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
  save('-v7', output_path, 'likelihood_times', 'smoother_times', 'loglik', ...
       'updated', 'filtered', 'state_names');
end
