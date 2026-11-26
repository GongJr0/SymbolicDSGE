% make_post82_kf_goldens.m
%   Run after:
%     dynare post82_kf.mod nograph
%
% Dumps the observations post82_kf.mod simulated, what Dynare's own smoother
% produced from them for the five variables our filter solves for, and the
% loglik Dynare's own kalman_filter.m assigns them.
%
% INDEXING. Dynare dates a prediction by the information set behind it rather
% than by the period being estimated, except for the one-step covariance. From
% the manual's filtered_vars entry, store_smoother_results.m, and
% missing_DiffuseKalmanSmootherH1_Z.m:
%
%   oo_.UpdatedVariables              a_{t|t}     dated by the period estimated
%   oo_.FilteredVariables             a_{t+1|t}   dated by the information set,
%                                                 so the array runs to one
%                                                 period past the sample
%   oo_.SmoothedVariables             a_{t|T}     dated by the period estimated
%   oo_.Smoother.Variance             P_{t|t-1}   dated by the period estimated,
%                                                 n x n x (T+1)
%   oo_.Smoother.State_uncertainty    P_{t|T}     n x n x T
%   oo_.Smoother.Init_State_uncertainty  P_{0|T}
%   oo_.SmoothedShocks                eta_{t|T}   one field per shock
%
% Our x_pred uses the other rule, so it holds the same predictions one row
% earlier. Nothing is shifted here. Reindexing is our side's problem and belongs
% in our tests, which is the same rule the decision-rule goldens follow.

  decl = {'g', 'z', 'r', 'x', 'Pi'};
  exo  = {'e_g', 'e_z', 'e_r'};
  n_decl = numel(decl);

  assert(isfield(oo_, 'UpdatedVariables'), ...
         'no smoother results: did dynare post82_kf.mod run?');
  assert(isfield(oo_, 'FilteredVariables'), ...
         'no filtered variables: calib_smoother needs the filtered_vars option');
  assert(isfield(oo_.Smoother, 'Variance'), ...
         'no one-step covariances: calib_smoother needs the filter_covariance option');
  assert(isfield(oo_.Smoother, 'State_uncertainty'), ...
         'no smoothed covariances: options_.smoothed_state_uncertainty was not set');

  endo_names = cellstr(M_.endo_names);
  idx_decl = cellfun(@(nm) find(strcmp(endo_names, nm), 1), decl);

  assert(max(abs(oo_.dr.ys(idx_decl))) < 1e-12, ...
         'a compared variable has a nonzero steady state; the dumps are not deviations');

  T_upd  = numel(oo_.UpdatedVariables.(decl{1}));
  T_filt = numel(oo_.FilteredVariables.(decl{1}));

  kf = struct();
  kf.decl_columns = decl;
  kf.exo_columns  = exo;
  kf.obs_columns  = kf_obs.name(:)';
  kf.obs          = kf_obs.data;

  kf.updated  = zeros(T_upd,  n_decl);
  kf.smoothed = zeros(T_upd,  n_decl);
  kf.filtered = zeros(T_filt, n_decl);
  for jj = 1:n_decl
    kf.updated(:, jj)  = oo_.UpdatedVariables.(decl{jj})(:);
    kf.smoothed(:, jj) = oo_.SmoothedVariables.(decl{jj})(:);
    kf.filtered(:, jj) = oo_.FilteredVariables.(decl{jj})(:);
  end

  kf.variance               = oo_.Smoother.Variance(idx_decl, idx_decl, :);
  kf.state_uncertainty      = oo_.Smoother.State_uncertainty(idx_decl, idx_decl, :);
  kf.init_state_uncertainty = oo_.Smoother.Init_State_uncertainty(idx_decl, idx_decl);

  kf.smoothed_shocks = zeros(T_upd, numel(exo));
  for jj = 1:numel(exo)
    kf.smoothed_shocks(:, jj) = oo_.SmoothedShocks.(exo{jj})(:);
  end

% ---------------------------------------------------------------------------
% LIKELIHOOD
% ---------------------------------------------------------------------------
% calib_smoother evaluates no likelihood, so kalman_filter.m runs on the same
% observations over the five variables above:
%     x_t = A x_{t-1} + R e_t,   e_t ~ N(0, Sigma_e)
%     y_t = d + Z x_t + w_t,     w_t ~ N(0, H)
% Z and d restate the OutGap, Infl and Rate equations of post82_kf.mod. A
% selection over all eight variables would need no restating, but under it
% P0 = I would also make OutGap independent of x, which is not our P0 = I.
%
% kalman_filter.m documents P as Var_0(alpha_1), the covariance the first
% observation updates, which is how our filter reads P0 as well. The
% unconditional P0 is the first slice of the smoother's own one-step
% covariances.

  assert(isequal(kf.obs_columns(:), options_.varobs(:)), ...
         'observation columns are not in varobs order');
  assert(all(ismember(oo_.dr.state_var, idx_decl)), ...
         'a state variable is outside the five compared ones');

  ghx_decl = zeros(M_.endo_nbr, numel(oo_.dr.state_var));
  ghu_decl = zeros(M_.endo_nbr, M_.exo_nbr);
  ghx_decl(oo_.dr.order_var, :) = oo_.dr.ghx;
  ghu_decl(oo_.dr.order_var, :) = oo_.dr.ghu;

  A_endo = zeros(M_.endo_nbr, M_.endo_nbr);
  A_endo(:, oo_.dr.state_var) = ghx_decl;
  A = A_endo(idx_decl, idx_decl);
  R = ghu_decl(idx_decl, :);

  param_names = cellstr(M_.param_names);
  pi_star = M_.params(strcmp(param_names, 'pi_star'));
  r_star  = M_.params(strcmp(param_names, 'r_star'));
  col = @(nm) find(strcmp(decl, nm), 1);

  Z = zeros(3, n_decl);
  Z(1, col('x'))  = 1.0;
  Z(2, col('Pi')) = 4.0;
  Z(3, col('r'))  = 4.0;
  d = [0.0; pi_star; r_star + pi_star];
  H = M_.H;

  assert(max(max(abs(d + Z * kf_path(idx_decl, 2:end) - kf.obs'))) < 1e-10, ...
         'Z and d do not reproduce the observations post82_kf.mod simulated');

  % riccati_tol = 0 keeps the full recursion for every period. At Dynare's
  % default the gain freezes once it stops moving.
  T_kf = size(kf.obs, 1);
  run_kf = @(P0) kalman_filter( ...
      kf.obs' - d, 1, T_kf, zeros(n_decl, 1), P0, ...
      1e-10, 0, false, 0, ...
      A, M_.Sigma_e, R, H, Z, ...
      n_decl, size(Z, 1), M_.exo_nbr, 1, 0, {}, false, false);

  % kalman_filter returns MINUS the log-likelihood, 2*pi constant included.
  kf.loglik_uncond = -run_kf(kf.variance(:, :, 1));
  kf.loglik_fixed  = -run_kf(eye(n_decl));

% ---------------------------------------------------------------------------
% SAVE
% ---------------------------------------------------------------------------

  save('-v7', '../goldens/post82_kf_goldens.mat', 'kf');

  printf('\n--- post82 kalman goldens, dynare smoother ---\n');
  printf('columns           : %s\n', strjoin(kf.decl_columns, ','));
  printf('observations      : %d rows (%s)\n', size(kf.obs, 1), strjoin(kf.obs_columns, ','));
  printf('updated           : %d rows (a_{t|t})\n',   size(kf.updated, 1));
  printf('filtered          : %d rows (a_{t+1|t})\n', size(kf.filtered, 1));
  printf('smoothed          : %d rows (a_{t|T})\n',   size(kf.smoothed, 1));
  printf('variance          : %d slices (P_{t|t-1})\n', size(kf.variance, 3));
  printf('state uncertainty : %d slices (P_{t|T})\n',   size(kf.state_uncertainty, 3));
  printf('smoothed shocks   : %d rows (%s)\n', size(kf.smoothed_shocks, 1), strjoin(kf.exo_columns, ','));
  printf('loglik, P0 uncond : %.17g\n', kf.loglik_uncond);
  printf('loglik, P0 = I    : %.17g\n', kf.loglik_fixed);
  printf('written: post82_kf_goldens.mat\n');
