% make_rbc_first_order_goldens.m
%   Run after:
%     dynare rbc_first_order.mod noclearall
%
% Produces three golden families at order 1:
%   1. decision rule  : ghx / ghu / ys, raw and in declaration order
%   2. simulations    : deterministic + stochastic paths, raw and sdsge-aligned
%   3. kalman filter  : loglik and filtered states on a single observable
%
% EVERY aligned view is dumped next to the raw Dynare object it came from. If
% our alignment turns out to be wrong, the raw arrays are enough to re-derive
% on the Python side without re-running Octave.
%
% CONVENTIONS UNDER TEST are stated where they are applied, and each one that
% has more than one defensible choice is dumped BOTH ways (P0, the 2*pi
% constant) so a mismatch never costs a round trip.

  iorder = 1;
  options_.pruning = 0;   % meaningless at order 1; off so it cannot bite

  endo_names  = cellstr(M_.endo_names);
  exo_names   = cellstr(M_.exo_names);
  param_names = cellstr(M_.param_names);

  idx_c = find(strcmp(endo_names, 'c'), 1);
  idx_k = find(strcmp(endo_names, 'k'), 1);
  idx_z = find(strcmp(endo_names, 'z'), 1);
  idx_e = find(strcmp(exo_names,  'e'), 1);

  rho = M_.params(find(strcmp(param_names, 'rho'), 1));
  sig = M_.params(find(strcmp(param_names, 'sig'), 1));

  n_endo = M_.endo_nbr;
  n_exo  = M_.exo_nbr;

  ys   = oo_.dr.ys;              % steady state, DECLARATION order
  k_ss = ys(idx_k);
  c_ss = ys(idx_c);

  sdsge_columns       = {'z', 'k', 'c'};   % our canonical var order
  dynare_decl_columns = endo_names;

% ---------------------------------------------------------------------------
% 1. DECISION RULE
% ---------------------------------------------------------------------------
% Dynare's ghx/ghu have rows in DR order (oo_.dr.order_var) and ghx columns in
% the order of oo_.dr.state_var (declaration indices of the states). Both index
% vectors are dumped so the mapping is reproducible from the raw arrays alone.

  ghx_raw    = oo_.dr.ghx;
  ghu_raw    = oo_.dr.ghu;
  order_var  = oo_.dr.order_var(:);
  state_var  = oo_.dr.state_var(:);
  Sigma_e    = M_.Sigma_e;

  % Rows into declaration order.
  ghx_decl = zeros(size(ghx_raw));
  ghu_decl = zeros(size(ghu_raw));
  ghx_decl(order_var, :) = ghx_raw;
  ghu_decl(order_var, :) = ghu_raw;

  % Full-vector state space in DECLARATION order:
  %   y_t - ys = A_decl * (y_{t-1} - ys) + R_decl * e_t
  % Columns of A_decl that are not state variables are structurally zero, which
  % is the Dynare-side analogue of our own zero right block.
  A_decl = zeros(n_endo, n_endo);
  A_decl(:, state_var) = ghx_decl;
  R_decl = ghu_decl;

% ---------------------------------------------------------------------------
% 2. SIMULATIONS
% ---------------------------------------------------------------------------
% Same x0 and shock sequence as the order-2 generator, so the two orders stay
% comparable to each other as well as to us.

  x0_sdsge = [0.02; 1.01 * k_ss];      % [z0; k0] in OUR convention

  T_det      = 24;
  shock_det  = zeros(T_det, 1);

  shock_stoch = sig * [
      0.25
     -0.70
      0.10
      0.60
     -0.20
      0.00
      0.35
     -0.45
      0.15
     -0.10
      0.05
      0.20
  ];
  T_stoch = length(shock_stoch);

  % --- THE ALIGNMENT ------------------------------------------------------
  % Our z-state leads Dynare's by one period, so seeding Dynare with z0/rho
  % makes Dynare's period-1 z equal our z0. Our k is the same predetermined
  % stock, so it seeds directly. This is the single conversion this whole
  % comparison rests on; it is applied in exactly one place per run.
  function y0 = seed_from_sdsge(ys, idx_z, idx_k, x0, rho)
    y0 = ys;
    y0(idx_z) = x0(1) / rho;
    y0(idx_k) = x0(2);
  end

  % Read a simult_ result back into our column order [z, k, c]. z and c are
  % read at the same columns; k is read one column earlier, because our k(t) is
  % the stock Dynare labels one period back.
  function out = to_sdsge(y, idx_z, idx_k, idx_c, T, k0)
    cols = 2:(T + 2);
    out = zeros(T + 1, 3);
    out(:, 1)     = y(idx_z, cols)';
    out(1, 2)     = k0;
    out(2:end, 2) = y(idx_k, cols(1:end-1))';
    out(:, 3)     = y(idx_c, cols)';
  end
  % ------------------------------------------------------------------------

  y0_det = seed_from_sdsge(ys, idx_z, idx_k, x0_sdsge, rho);
  ex_det = zeros(T_det + 1, n_exo);
  ex_det(2:end, idx_e) = shock_det(:);
  try
      y_det = simult_(M_, options_, y0_det, oo_.dr, ex_det, iorder);
  catch
      y_det = simult_(y0_det, oo_.dr, ex_det, iorder);
  end
  det_sdsge      = to_sdsge(y_det, idx_z, idx_k, idx_c, T_det, x0_sdsge(2));
  det_dynare_raw = y_det(:, 2:end)';

  y0_stoch = seed_from_sdsge(ys, idx_z, idx_k, x0_sdsge, rho);
  ex_stoch = zeros(T_stoch + 1, n_exo);
  ex_stoch(2:end, idx_e) = shock_stoch(:);
  try
      y_stoch = simult_(M_, options_, y0_stoch, oo_.dr, ex_stoch, iorder);
  catch
      y_stoch = simult_(y0_stoch, oo_.dr, ex_stoch, iorder);
  end
  stoch_sdsge      = to_sdsge(y_stoch, idx_z, idx_k, idx_c, T_stoch, x0_sdsge(2));
  stoch_dynare_raw = y_stoch(:, 2:end)';

  T_irf        = 24;
  x0_irf_sdsge = [0; k_ss];
  shock_irf    = zeros(T_irf, 1);
  shock_irf(1) = sig;

  y0_irf   = seed_from_sdsge(ys, idx_z, idx_k, x0_irf_sdsge, rho);
  ex_irf   = zeros(T_irf + 1, n_exo);
  ex_irf(2, idx_e) = sig;
  ex_base  = zeros(T_irf + 1, n_exo);
  try
      y_irf      = simult_(M_, options_, y0_irf, oo_.dr, ex_irf,  iorder);
      y_irf_base = simult_(M_, options_, y0_irf, oo_.dr, ex_base, iorder);
  catch
      y_irf      = simult_(y0_irf, oo_.dr, ex_irf,  iorder);
      y_irf_base = simult_(y0_irf, oo_.dr, ex_base, iorder);
  end
  irf_level_sdsge = to_sdsge(y_irf,      idx_z, idx_k, idx_c, T_irf, x0_irf_sdsge(2));
  irf_base_sdsge  = to_sdsge(y_irf_base, idx_z, idx_k, idx_c, T_irf, x0_irf_sdsge(2));
  irf_sdsge       = irf_level_sdsge - irf_base_sdsge;
  irf_dynare_raw  = y_irf(:, 2:end)' - y_irf_base(:, 2:end)';

% ---------------------------------------------------------------------------
% 3. KALMAN FILTER
% ---------------------------------------------------------------------------
% State space (declaration order, deviations from ys):
%     x_t = A_decl x_{t-1} + R_decl e_t,   e_t ~ N(0, Sigma_e)
%     d_t = Z x_t                          (no measurement error)
%
% TIMING UNDER TEST: the observation at t is paired with the state at t, i.e.
% predict x_{t|t-1} = A x_{t-1|t-1} FIRST, then update with d_t. A one-period
% misalignment changes the loglik by far more than any tolerance, which is the
% whole reason this golden is worth more than a matrix comparison.
%
% One observable (c) and one shock, so F is scalar and nonsingular with no
% measurement error. Data are generated from the stochastic path above and
% DUMPED, so both sides filter byte-identical numbers rather than regenerating.

  Z = zeros(1, n_endo);
  Z(idx_c) = 1.0;
  H = 0.0;

  % Observable in deviation from steady state, aligned to our row convention:
  % kf_data(i) is the observation for our x_i, i = 1..T_stoch.
  kf_data = stoch_sdsge(2:end, 3) - c_ss;
  T_kf    = length(kf_data);

  % --- P0, both defensible choices ---------------------------------------
  % (a) unconditional covariance, Dynare's default flavour, by doubling:
  %     P = sum_j A^j R Q R' A'^j. No control-package dependency.
  Q  = Sigma_e;
  RQ = R_decl * Q * R_decl';
  P_uncond = RQ;
  Ak = A_decl;
  for iter = 1:60
    P_uncond = P_uncond + Ak * P_uncond * Ak';
    Ak = Ak * Ak;
  end
  P_uncond = 0.5 * (P_uncond + P_uncond');   % symmetrize drift

  % (b) a fixed diagonal, reproducible with no solver at all.
  P_fixed = 1e-2 * eye(n_endo);
  % ------------------------------------------------------------------------

  function [ll_const, ll_noconst, xf, xp] = run_kf(A, R, Q, Z, H, P0, data)
    n  = size(A, 1);
    T  = length(data);
    x  = zeros(n, 1);
    P  = P0;
    xf = zeros(T, n);
    xp = zeros(T, n);
    ll_const   = 0.0;
    ll_noconst = 0.0;
    RQR = R * Q * R';
    for tt = 1:T
      % predict
      x = A * x;
      P = A * P * A' + RQR;
      xp(tt, :) = x';
      % update
      v = data(tt) - Z * x;
      F = Z * P * Z' + H;
      K = (P * Z') / F;
      x = x + K * v;
      P = P - K * (Z * P);
      P = 0.5 * (P + P');
      xf(tt, :) = x';
      quad = log(det(F)) + (v' / F) * v;
      ll_noconst = ll_noconst - 0.5 * quad;
      ll_const   = ll_const   - 0.5 * (quad + length(v) * log(2 * pi));
    end
  end

  [kf_loglik_const_uncond, kf_loglik_noconst_uncond, ...
   kf_filtered_uncond, kf_predicted_uncond] = ...
      run_kf(A_decl, R_decl, Q, Z, H, P_uncond, kf_data);

  [kf_loglik_const_fixed, kf_loglik_noconst_fixed, ...
   kf_filtered_fixed, kf_predicted_fixed] = ...
      run_kf(A_decl, R_decl, Q, Z, H, P_fixed, kf_data);

  % Filtered states in our column order [z, k, c]. These are DECLARATION-order
  % states from the filter, so only the column permutation applies; there is no
  % k shift here, because the filter's x_t is a single dated vector rather than
  % a simulated path.
  perm_sdsge = [idx_z, idx_k, idx_c];
  kf_filtered_uncond_sdsge = kf_filtered_uncond(:, perm_sdsge);
  kf_filtered_fixed_sdsge  = kf_filtered_fixed(:,  perm_sdsge);

% ---------------------------------------------------------------------------
% SAVE
% ---------------------------------------------------------------------------

  save('-v7', 'rbc_first_order_goldens.mat', ...
       'ghx_raw', 'ghu_raw', 'ghx_decl', 'ghu_decl', ...
       'order_var', 'state_var', 'ys', 'Sigma_e', ...
       'A_decl', 'R_decl', ...
       'x0_sdsge', 'shock_det', 'shock_stoch', 'x0_irf_sdsge', 'shock_irf', ...
       'det_sdsge', 'stoch_sdsge', 'irf_sdsge', ...
       'irf_level_sdsge', 'irf_base_sdsge', ...
       'det_dynare_raw', 'stoch_dynare_raw', 'irf_dynare_raw', ...
       'kf_data', 'P_uncond', 'P_fixed', 'Z', 'H', ...
       'kf_loglik_const_uncond', 'kf_loglik_noconst_uncond', ...
       'kf_loglik_const_fixed', 'kf_loglik_noconst_fixed', ...
       'kf_filtered_uncond', 'kf_filtered_fixed', ...
       'kf_filtered_uncond_sdsge', 'kf_filtered_fixed_sdsge', ...
       'kf_predicted_uncond', 'kf_predicted_fixed', ...
       'sdsge_columns', 'dynare_decl_columns');

  dump = @(name, M) dlmwrite(name, M, 'precision', '%.17g');

  dump('rbc_fo_ghx_decl.csv',        ghx_decl);
  dump('rbc_fo_ghu_decl.csv',        ghu_decl);
  dump('rbc_fo_A_decl.csv',          A_decl);
  dump('rbc_fo_R_decl.csv',          R_decl);
  dump('rbc_fo_ys.csv',              ys);
  dump('rbc_fo_state_var.csv',       state_var);
  dump('rbc_fo_order_var.csv',       order_var);
  dump('rbc_fo_det_sdsge.csv',       det_sdsge);
  dump('rbc_fo_stoch_sdsge.csv',     stoch_sdsge);
  dump('rbc_fo_irf_sdsge.csv',       irf_sdsge);
  dump('rbc_fo_kf_data.csv',         kf_data);
  dump('rbc_fo_kf_P_uncond.csv',     P_uncond);
  dump('rbc_fo_kf_filtered_uncond.csv', kf_filtered_uncond_sdsge);
  dump('rbc_fo_kf_filtered_fixed.csv',  kf_filtered_fixed_sdsge);
  dump('rbc_fo_kf_loglik.csv', [kf_loglik_const_uncond, kf_loglik_noconst_uncond, ...
                                kf_loglik_const_fixed,  kf_loglik_noconst_fixed]);

  printf('\n--- rbc first order goldens ---\n');
  printf('ys (decl order %s): %.17g %.17g %.17g\n', ...
         strjoin(dynare_decl_columns', ','), ys(1), ys(2), ys(3));
  printf('state_var: %s\n', mat2str(state_var'));
  printf('loglik  uncond P0: const %.17g   noconst %.17g\n', ...
         kf_loglik_const_uncond, kf_loglik_noconst_uncond);
  printf('loglik  fixed  P0: const %.17g   noconst %.17g\n', ...
         kf_loglik_const_fixed, kf_loglik_noconst_fixed);
  printf('written: rbc_first_order_goldens.mat + rbc_fo_*.csv\n');
