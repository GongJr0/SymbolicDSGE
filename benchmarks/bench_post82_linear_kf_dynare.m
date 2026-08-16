function bench_post82_linear_kf_dynare(workdir, warmup, reps, output_path)
% Benchmark Dynare's POST82 likelihood and filter plus smoother entries.
%
% Initialization calls the public `dynare post82_kf.mod` entry once. The timed
% closure then calls Dynare's `kalman_filter` directly, as Dynare has no
% filter only model file command. It also times the generated
% `calib_smoother` entry through `evaluate_smoother`, which materializes Dynare
% filter and smoother outputs. All transition and measurement matrices come
% from the initialized Dynare model. Input data are prepared before this
% function is called and are not part of either timed loop.

  if nargin ~= 4
    error('Expected workdir, warmup, reps, and output_path.');
  end

  previous_dir = pwd;
  restore_dir = onCleanup(@() cd(previous_dir));
  cd(workdir);

  % This is setup only. `noclearall` retains the session for the timed calls.
  dynare post82_kf.mod noclearall nograph;

  global M_ options_ oo_ estim_params_ bayestopt_
  endo_names = cellstr(M_.endo_names);
  param_names = cellstr(M_.param_names);
  n_endo = M_.endo_nbr;
  n_exo = M_.exo_nbr;

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

  pi_star = M_.params(find(strcmp(param_names, 'pi_star'), 1));
  r_star = M_.params(find(strcmp(param_names, 'r_star'), 1));
  idx_r = find(strcmp(endo_names, 'r'), 1);
  idx_x = find(strcmp(endo_names, 'x'), 1);
  idx_pi = find(strcmp(endo_names, 'Pi'), 1);
  Z = zeros(3, n_endo);
  Z(1, idx_x) = 1.0;
  Z(2, idx_pi) = 4.0;
  Z(3, idx_r) = 4.0;
  d = [0.0; pi_star; r_star + pi_star];
  H = eye(3);

  run('post82_kf_data.m');
  Y = [OutGap, Infl, Rate]' - d * ones(1, numel(OutGap));
  periods = size(Y, 2);

  % `kalman_filter` requires an explicit P0. The full endogenous-state
  % realization used here is larger than Dynare's reduced state transition,
  % so derive its stationary covariance directly.
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
      n_endo, size(Z, 1), n_exo, 1, 0, {}, false, false);

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

  var_list_ = {};
  for i = 1:warmup
    [oo_, M_, options_, bayestopt_] = evaluate_smoother( ...
        options_.parameter_set, var_list_, M_, oo_, options_, bayestopt_, estim_params_);
  end
  smoother_times = zeros(reps, 1);
  for i = 1:reps
    started = tic;
    [oo_, M_, options_, bayestopt_] = evaluate_smoother( ...
        options_.parameter_set, var_list_, M_, oo_, options_, bayestopt_, estim_params_);
    smoother_times(i) = toc(started);
  end
  loglik = -negloglik;

  state_names = {'g', 'z', 'r', 'x', 'Pi'};
  updated = zeros(numel(oo_.UpdatedVariables.(state_names{1})), numel(state_names));
  filtered = zeros(numel(oo_.FilteredVariables.(state_names{1})), numel(state_names));
  for i = 1:numel(state_names)
    updated(:, i) = oo_.UpdatedVariables.(state_names{i})(:);
    filtered(:, i) = oo_.FilteredVariables.(state_names{i})(:);
  end
  save('-v7', output_path, 'likelihood_times', 'smoother_times', 'loglik', ...
       'updated', 'filtered');
end
