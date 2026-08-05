% make_rbc_multishock_second_order_sim_goldens.m
%   Run after:
%     dynare rbc_multishock_second_order.mod noclearall
%     make_rbc_multishock_second_order_sim_goldens
%
% The three-shock twin of make_rbc_second_order_sim_goldens.m. That one pins the
% order-2 simulator against a single innovation, where the lifted shock state
% means ghxu and ghuu are already inside the ghxx column being contracted. Here
% the innovations are distinct and simultaneously nonzero, so the cross terms
% ghuu(e_i, e_j) and ghxu(x, e_j) with i != j reach the state update. Nothing
% else in the suite contracts those.
%
% Same conventions as the single-shock script, so the two are readable against
% each other: pruned second order, columns stored in our layout with k dated as
% the predetermined stock, and the path seeded one AR step behind the printed
% z/d/g so that the first reported row is the initial condition. Flat style, no
% local functions, for the same reason.

  options_.pruning = 1;
  iorder = 2;

  endo_names  = cellstr(M_.endo_names);
  exo_names   = cellstr(M_.exo_names);
  param_names = cellstr(M_.param_names);

  idx_c = find(strcmp(endo_names, 'c'), 1);
  idx_k = find(strcmp(endo_names, 'k'), 1);
  idx_z = find(strcmp(endo_names, 'z'), 1);
  idx_d = find(strcmp(endo_names, 'd'), 1);
  idx_g = find(strcmp(endo_names, 'g'), 1);

  idx_e_z = find(strcmp(exo_names, 'e_z'), 1);
  idx_e_d = find(strcmp(exo_names, 'e_d'), 1);
  idx_e_g = find(strcmp(exo_names, 'e_g'), 1);

  rho_z = M_.params(find(strcmp(param_names, 'rho_z'), 1));
  rho_d = M_.params(find(strcmp(param_names, 'rho_d'), 1));
  rho_g = M_.params(find(strcmp(param_names, 'rho_g'), 1));
  sig_z = M_.params(find(strcmp(param_names, 'sig_z'), 1));
  sig_d = M_.params(find(strcmp(param_names, 'sig_d'), 1));
  sig_g = M_.params(find(strcmp(param_names, 'sig_g'), 1));

  k_ss = oo_.dr.ys(idx_k);

  % Our column order. 'k' here is the predetermined stock, which is k_lag1 on
  % our side, not the contemporaneous k.
  sdsge_columns = {'z', 'd', 'g', 'k', 'c'};

  % [z, d, g, k]: the three processes contemporaneously, then capital.
  x0_sdsge = [0.02; -0.015; 0.03; 1.01 * k_ss];

  % --- deterministic: displaced initial state, no innovations ----------------
  % Shock cross terms are dead here by construction. What this pins is hxx/gxx
  % on a four-state model plus the correlated risk correction, which the pruned
  % recursion applies every period whether or not an innovation lands.
  T_det = 24;
  shock_det = zeros(T_det, M_.exo_nbr);

  % simult_ takes the period-0 level vector, and one AR step turns the seed into
  % the value we mean, so each process goes in divided by its own rho.
  y0_det = oo_.dr.ys;
  y0_det(idx_z) = x0_sdsge(1) / rho_z;
  y0_det(idx_d) = x0_sdsge(2) / rho_d;
  y0_det(idx_g) = x0_sdsge(3) / rho_g;
  y0_det(idx_k) = x0_sdsge(4);

  ex_det = [zeros(1, M_.exo_nbr); shock_det];

  try
      y_det = simult_(M_, options_, y0_det, oo_.dr, ex_det, iorder);
  catch
      y_det = simult_(y0_det, oo_.dr, ex_det, iorder);
  end

  % Columns [z, d, g, k, c]. k is lagged by one period, which is what makes it
  % our k_lag1 rather than our contemporaneous k.
  det_sdsge = zeros(T_det + 1, 5);
  sim_cols = 2:(T_det + 2);
  det_sdsge(:, 1) = y_det(idx_z, sim_cols)';
  det_sdsge(:, 2) = y_det(idx_d, sim_cols)';
  det_sdsge(:, 3) = y_det(idx_g, sim_cols)';
  det_sdsge(1, 4) = x0_sdsge(4);
  det_sdsge(2:end, 4) = y_det(idx_k, sim_cols(1:end - 1))';
  det_sdsge(:, 5) = y_det(idx_c, sim_cols)';

  % --- stochastic: all three innovations live in the same period -------------
  % The point of the fixture. Every row has at least two nonzero innovations, so
  % the ghuu off-diagonals contribute to every state update.
  shock_stoch_std = [
      0.25  -0.60   0.40
     -0.70   0.35  -0.15
      0.10   0.80   0.55
      0.60  -0.25  -0.70
     -0.20   0.15   0.30
      0.00  -0.45   0.20
      0.35   0.50  -0.35
     -0.45  -0.10   0.65
      0.15   0.70  -0.20
     -0.10  -0.55   0.10
      0.05   0.20   0.45
      0.20   0.30  -0.60
  ];
  % Raw innovations, not standardized draws: simult_ takes ex_ in levels and our
  % B is a plain selector, so the same numbers go into both sides untouched.
  shock_stoch = shock_stoch_std * diag([sig_z, sig_d, sig_g]);
  T_stoch = size(shock_stoch, 1);

  y0_stoch = oo_.dr.ys;
  y0_stoch(idx_z) = x0_sdsge(1) / rho_z;
  y0_stoch(idx_d) = x0_sdsge(2) / rho_d;
  y0_stoch(idx_g) = x0_sdsge(3) / rho_g;
  y0_stoch(idx_k) = x0_sdsge(4);

  ex_stoch = [zeros(1, M_.exo_nbr); shock_stoch];

  try
      y_stoch = simult_(M_, options_, y0_stoch, oo_.dr, ex_stoch, iorder);
  catch
      y_stoch = simult_(y0_stoch, oo_.dr, ex_stoch, iorder);
  end

  stoch_sdsge = zeros(T_stoch + 1, 5);
  sim_cols = 2:(T_stoch + 2);
  stoch_sdsge(:, 1) = y_stoch(idx_z, sim_cols)';
  stoch_sdsge(:, 2) = y_stoch(idx_d, sim_cols)';
  stoch_sdsge(:, 3) = y_stoch(idx_g, sim_cols)';
  stoch_sdsge(1, 4) = x0_sdsge(4);
  stoch_sdsge(2:end, 4) = y_stoch(idx_k, sim_cols(1:end - 1))';
  stoch_sdsge(:, 5) = y_stoch(idx_c, sim_cols)';

  % --- IRF: one-standard-deviation impulse in all three at once --------------
  % Our irf() puts each shock's own sigma in the impact period and differences
  % against a no-shock baseline, which is what the two runs below reproduce.
  % Firing all three together is deliberate: a single-shock IRF would leave the
  % ghuu off-diagonals at zero again.
  T_irf = 24;
  x0_irf_sdsge = [0; 0; 0; k_ss];

  y0_irf = oo_.dr.ys;
  y0_irf(idx_k) = x0_irf_sdsge(4);

  ex_irf = zeros(T_irf + 1, M_.exo_nbr);
  ex_irf(2, idx_e_z) = sig_z;
  ex_irf(2, idx_e_d) = sig_d;
  ex_irf(2, idx_e_g) = sig_g;
  ex_irf_base = zeros(T_irf + 1, M_.exo_nbr);

  try
      y_irf      = simult_(M_, options_, y0_irf, oo_.dr, ex_irf, iorder);
      y_irf_base = simult_(M_, options_, y0_irf, oo_.dr, ex_irf_base, iorder);
  catch
      y_irf      = simult_(y0_irf, oo_.dr, ex_irf, iorder);
      y_irf_base = simult_(y0_irf, oo_.dr, ex_irf_base, iorder);
  end

  irf_level_sdsge = zeros(T_irf + 1, 5);
  irf_base_sdsge  = zeros(T_irf + 1, 5);
  sim_cols = 2:(T_irf + 2);
  irf_level_sdsge(:, 1) = y_irf(idx_z, sim_cols)';
  irf_level_sdsge(:, 2) = y_irf(idx_d, sim_cols)';
  irf_level_sdsge(:, 3) = y_irf(idx_g, sim_cols)';
  irf_level_sdsge(1, 4) = x0_irf_sdsge(4);
  irf_level_sdsge(2:end, 4) = y_irf(idx_k, sim_cols(1:end - 1))';
  irf_level_sdsge(:, 5) = y_irf(idx_c, sim_cols)';
  irf_base_sdsge(:, 1) = y_irf_base(idx_z, sim_cols)';
  irf_base_sdsge(:, 2) = y_irf_base(idx_d, sim_cols)';
  irf_base_sdsge(:, 3) = y_irf_base(idx_g, sim_cols)';
  irf_base_sdsge(1, 4) = x0_irf_sdsge(4);
  irf_base_sdsge(2:end, 4) = y_irf_base(idx_k, sim_cols(1:end - 1))';
  irf_base_sdsge(:, 5) = y_irf_base(idx_c, sim_cols)';
  irf_sdsge = irf_level_sdsge - irf_base_sdsge;

% ---------------------------------------------------------------------------
% Paste block, same shape as the decision-rule script: Python literals, one list
% per period, so the test file carries them without a data file.
% ---------------------------------------------------------------------------

  fprintf('\n--- rbc_multishock second order sim goldens ---\n');
  fprintf('columns     : %s\n', strjoin(sdsge_columns, ','));
  fprintf('exo   order : %s\n', strjoin(exo_names', ','));
  fprintf('x0 [z,d,g,k]: %.17g, %.17g, %.17g, %.17g\n', x0_sdsge);
  fprintf('\n');

  names = {'MS_STOCHASTIC_SHOCKS', 'MS_DETERMINISTIC_SIM', ...
           'MS_STOCHASTIC_SIM', 'MS_IRF'};
  mats  = {shock_stoch, det_sdsge, stoch_sdsge, irf_sdsge};
  for m = 1:numel(names)
    A = mats{m};
    fprintf('_DYNARE_%s = np.array(\n    [\n', names{m});
    for i = 1:size(A, 1)
      fprintf('        [');
      fprintf('%.17g, ', A(i, 1:end - 1));
      fprintf('%.17g],\n', A(i, end));
    end
    fprintf('    ],\n    dtype=np.float64,\n)\n\n');
  end

  fprintf('_DYNARE_MS_SIM_X0 = (%.17g, %.17g, %.17g, %.17g)\n', x0_sdsge);
