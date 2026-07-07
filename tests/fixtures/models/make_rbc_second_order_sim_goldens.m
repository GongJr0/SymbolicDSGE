% make_rbc_second_order_sim_goldens.m
  % Run after:
  %   dynare rbc_second_order.mod noclearall

  options_.pruning = 1;
  iorder = 2;

  endo_names = cellstr(M_.endo_names);
  exo_names = cellstr(M_.exo_names);
  param_names = cellstr(M_.param_names);

  idx_c = find(strcmp(endo_names, 'c'), 1);
  idx_k = find(strcmp(endo_names, 'k'), 1);
  idx_z = find(strcmp(endo_names, 'z'), 1);
  idx_e = find(strcmp(exo_names, 'e'), 1);
  idx_rho = find(strcmp(param_names, 'rho'), 1);
  idx_sig = find(strcmp(param_names, 'sig'), 1);

  rho = M_.params(idx_rho);
  sig = M_.params(idx_sig);
  k_ss = oo_.dr.ys(idx_k);

  sdsge_columns = {'z', 'k', 'c'};
  dynare_decl_columns = endo_names;

  x0_sdsge = [0.02; 1.01 * k_ss];

  T_det = 24;
  shock_det = zeros(T_det, 1);

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

  y0_det = oo_.dr.ys;
  y0_det(idx_z) = x0_sdsge(1) / rho;
  y0_det(idx_k) = x0_sdsge(2);

  ex_det = zeros(T_det + 1, M_.exo_nbr);
  ex_det(2:end, idx_e) = shock_det(:);

  try
      y_det = simult_(M_, options_, y0_det, oo_.dr, ex_det, iorder);
  catch
      y_det = simult_(y0_det, oo_.dr, ex_det, iorder);
  end

  det_sdsge = zeros(T_det + 1, 3);
  sim_cols = 2:(T_det + 2);
  det_sdsge(:, 1) = y_det(idx_z, sim_cols)';
  det_sdsge(1, 2) = x0_sdsge(2);
  det_sdsge(2:end, 2) = y_det(idx_k, sim_cols(1:end-1))';
  det_sdsge(:, 3) = y_det(idx_c, sim_cols)';
  det_dynare_decl = y_det(:, 2:end)';

  T_stoch = length(shock_stoch);

  y0_stoch = oo_.dr.ys;
  y0_stoch(idx_z) = x0_sdsge(1) / rho;
  y0_stoch(idx_k) = x0_sdsge(2);

  ex_stoch = zeros(T_stoch + 1, M_.exo_nbr);
  ex_stoch(2:end, idx_e) = shock_stoch(:);

  try
      y_stoch = simult_(M_, options_, y0_stoch, oo_.dr, ex_stoch, iorder);
  catch
      y_stoch = simult_(y0_stoch, oo_.dr, ex_stoch, iorder);
  end

  stoch_sdsge = zeros(T_stoch + 1, 3);
  sim_cols = 2:(T_stoch + 2);
  stoch_sdsge(:, 1) = y_stoch(idx_z, sim_cols)';
  stoch_sdsge(1, 2) = x0_sdsge(2);
  stoch_sdsge(2:end, 2) = y_stoch(idx_k, sim_cols(1:end-1))';
  stoch_sdsge(:, 3) = y_stoch(idx_c, sim_cols)';
  stoch_dynare_decl = y_stoch(:, 2:end)';

  save('-v7', 'rbc_second_order_sim_goldens.mat', ...
       'x0_sdsge', 'shock_det', 'shock_stoch', ...
       'det_sdsge', 'stoch_sdsge', ...
       'det_dynare_decl', 'stoch_dynare_decl', ...
       'sdsge_columns', 'dynare_decl_columns');

  dlmwrite('rbc_second_order_det_sdsge.csv', det_sdsge, 'precision', '%.17g');
  dlmwrite('rbc_second_order_stoch_sdsge.csv', stoch_sdsge, 'precision', '%.17g');
