% make_post82_first_order_goldens.m
%   Run after:
%     dynare post82_first_order.mod noclearall
%
% Produces three golden families at order 1:
%   1. decision rule  : ghx / ghu / ys
%   2. simulations    : deterministic, stochastic, and one IRF per shock
%   3. kalman filter  : total and per-period loglik, three observables
%
% The filter is Dynare's own kalman_filter.m, called directly. Nothing in this
% file reimplements a recursion: a golden produced by our own MATLAB would only
% establish that we can write the same loop twice.
%
% State paths are not here. They come from Dynare's smoother, which needs a
% varobs block, so they live in post82_kf.mod and make_post82_kf_goldens.m. This
% script writes post82_kf_data.m, the datafile that pair reads, so both halves
% filter byte-identical numbers.
%
% RAW ONLY. Nothing here re-dates, rescales, or re-indexes a Dynare output to
% suit our conventions. Rows are permuted from DR order into DECLARATION order,
% which is labelling rather than conversion, and order_var / state_var are dumped
% so even that is reproducible from the raw arrays. Any alignment our side needs
% is our side's problem and belongs in our tests, not in this file.

  iorder = 1;
  options_.pruning = 0;   % meaningless at order 1; off so it cannot bite

  endo_names  = cellstr(M_.endo_names);
  exo_names   = cellstr(M_.exo_names);
  param_names = cellstr(M_.param_names);

  idx_g  = find(strcmp(endo_names, 'g'),  1);
  idx_z  = find(strcmp(endo_names, 'z'),  1);
  idx_r  = find(strcmp(endo_names, 'r'),  1);
  idx_x  = find(strcmp(endo_names, 'x'),  1);
  idx_Pi = find(strcmp(endo_names, 'Pi'), 1);

  idx_eg = find(strcmp(exo_names, 'e_g'), 1);
  idx_ez = find(strcmp(exo_names, 'e_z'), 1);
  idx_er = find(strcmp(exo_names, 'e_r'), 1);

  getp    = @(nm) M_.params(find(strcmp(param_names, nm), 1));
  pi_star = getp('pi_star');
  r_star  = getp('r_star');

  n_endo = M_.endo_nbr;
  n_exo  = M_.exo_nbr;

  ys = oo_.dr.ys;   % all zeros: every variable is a gap

  dynare_decl_columns = endo_names;
  dynare_exo_columns  = exo_names;

% ---------------------------------------------------------------------------
% 1. DECISION RULE
% ---------------------------------------------------------------------------
% ghx/ghu rows are in DR order (oo_.dr.order_var); ghx columns follow
% oo_.dr.state_var. Both index vectors are dumped, so the declaration-order
% views below are reproducible from the raw arrays alone.

  ghx_raw   = oo_.dr.ghx;
  ghu_raw   = oo_.dr.ghu;
  order_var = oo_.dr.order_var(:);
  state_var = oo_.dr.state_var(:);
  Sigma_e   = M_.Sigma_e;

  ghx_decl = zeros(size(ghx_raw));
  ghu_decl = zeros(size(ghu_raw));
  ghx_decl(order_var, :) = ghx_raw;
  ghu_decl(order_var, :) = ghu_raw;

  % Full-vector state space in DECLARATION order:
  %   y_t - ys = A_decl * (y_{t-1} - ys) + R_decl * e_t
  A_decl = zeros(n_endo, n_endo);
  A_decl(:, state_var) = ghx_decl;
  R_decl = ghu_decl;

  % The single number issue #390 turns on: the impact of e_r on r. The Taylor
  % rule carries Pi and x at t, so this is a fixed point rather than one.
  impact_r_er = R_decl(idx_r, idx_er);

% ---------------------------------------------------------------------------
% 2. SIMULATIONS
% ---------------------------------------------------------------------------
% simult_ takes y0 as the full endogenous vector one period before the first
% simulated period, and reads the state components off it. Deviations are placed
% on g, z and r; x and Pi are left at ys because they are not free.
%
% simult_ simulates one period per ROW of the shock matrix, and returns y0 in
% column 1 followed by those periods, so ex_(k,:) drives period k. Dropping
% column 1 leaves x1 onward, which is what Dynare itself reports: endo_simul
% omits the initial condition and oo_.irfs starts at impact. Every path below
% is therefore the same length as the shock block that drove it, and carries no
% pre-impact row.

  x0_states = [0.5; -0.3; 0.2];   % [g; z; r], deviations from ys

  y0 = ys;
  y0(idx_g) = ys(idx_g) + x0_states(1);
  y0(idx_z) = ys(idx_z) + x0_states(2);
  y0(idx_r) = ys(idx_r) + x0_states(3);

  T_sim = 12;

  % Fixed innovation block, columns [e_g, e_z, e_r] in DECLARATION order of the
  % exogenous block. Hardcoded so both sides filter byte-identical numbers.
  shock_block = [
     0.25  -0.10   0.40
    -0.70   0.55  -0.15
     0.10   0.20   0.00
     0.60  -0.35   0.25
    -0.20   0.05  -0.50
     0.00   0.30   0.10
     0.35  -0.45   0.20
    -0.45   0.15  -0.30
     0.15   0.60   0.05
    -0.10  -0.25   0.35
     0.05   0.40  -0.20
     0.20  -0.05   0.15
  ];

  ex_det = zeros(T_sim, n_exo);
  try
      y_det = simult_(M_, options_, y0, oo_.dr, ex_det, iorder);
  catch
      y_det = simult_(y0, oo_.dr, ex_det, iorder);
  end
  det_dynare = y_det(:, 2:end)';   % (T_sim, n_endo), declaration order

  ex_stoch = zeros(T_sim, n_exo);
  ex_stoch(:, idx_eg) = shock_block(:, 1);
  ex_stoch(:, idx_ez) = shock_block(:, 2);
  ex_stoch(:, idx_er) = shock_block(:, 3);
  try
      y_stoch = simult_(M_, options_, y0, oo_.dr, ex_stoch, iorder);
  catch
      y_stoch = simult_(y0, oo_.dr, ex_stoch, iorder);
  end
  stoch_dynare = y_stoch(:, 2:end)';

  % One unit-innovation IRF per shock, from the steady state. Differencing
  % against a zero-shock run from the same y0 keeps this a pure response even
  % though the model is linear. Row 1 is the impact response, matching the
  % layout of oo_.irfs, so it is by definition a column of ghu.
  T_irf    = 16;
  irf_unit = zeros(T_irf, n_endo, n_exo);
  for jj = 1:n_exo
    ex_one = zeros(T_irf, n_exo);
    ex_one(1, jj) = 1.0;
    ex_nil = zeros(T_irf, n_exo);
    try
        y_one = simult_(M_, options_, ys, oo_.dr, ex_one, iorder);
        y_nil = simult_(M_, options_, ys, oo_.dr, ex_nil, iorder);
    catch
        y_one = simult_(ys, oo_.dr, ex_one, iorder);
        y_nil = simult_(ys, oo_.dr, ex_nil, iorder);
    end
    irf_unit(:, :, jj) = y_one(:, 2:end)' - y_nil(:, 2:end)';
  end
  irf_eg = irf_unit(:, :, idx_eg);
  irf_ez = irf_unit(:, :, idx_ez);
  irf_er = irf_unit(:, :, idx_er);

  % Guards on the row convention stated above: simult_ returns y0 in column 1
  % and one period per row of the shock matrix after it, so ex_(k,:) drives
  % row k. Row 1 of an IRF is then the impact response, which is the matching
  % column of R_decl. If either trips, every path in this file is off by one
  % and the goldens are not worth transcribing.
  assert(size(stoch_dynare, 1) == size(shock_block, 1), ...
         'path is not the same length as the shock block that drove it');
  for jj = 1:n_exo
    assert(max(abs(irf_unit(1, :, jj)' - R_decl(:, jj))) < 1e-10, ...
           'IRF row 1 does not reproduce R_decl column %d', jj);
  end

% ---------------------------------------------------------------------------
% SAVE
% ---------------------------------------------------------------------------

  save('-v7', 'post82_first_order_goldens.mat', ...
       'ghx_raw', 'ghu_raw', 'ghx_decl', 'ghu_decl', ...
       'order_var', 'state_var', 'ys', 'Sigma_e', ...
       'A_decl', 'R_decl', 'impact_r_er', ...
       'x0_states', 'shock_block', ...
       'det_dynare', 'stoch_dynare', ...
       'irf_eg', 'irf_ez', 'irf_er', ...
       'dynare_decl_columns', 'dynare_exo_columns');

  dump = @(name, M) dlmwrite(name, M, 'precision', '%.17g');

  dump('post82_fo_ghx_raw.csv',   ghx_raw);
  dump('post82_fo_ghu_raw.csv',   ghu_raw);
  dump('post82_fo_ghx_decl.csv',  ghx_decl);
  dump('post82_fo_ghu_decl.csv',  ghu_decl);
  dump('post82_fo_A_decl.csv',    A_decl);
  dump('post82_fo_R_decl.csv',    R_decl);
  dump('post82_fo_ys.csv',        ys);
  dump('post82_fo_Sigma_e.csv',   Sigma_e);
  dump('post82_fo_order_var.csv', order_var);
  dump('post82_fo_state_var.csv', state_var);
  dump('post82_fo_det.csv',       det_dynare);
  dump('post82_fo_stoch.csv',     stoch_dynare);
  dump('post82_fo_irf_eg.csv',    irf_eg);
  dump('post82_fo_irf_ez.csv',    irf_ez);
  dump('post82_fo_irf_er.csv',    irf_er);

  printf('\n--- post82 first order goldens ---\n');
  printf('decl order: %s\n', strjoin(dynare_decl_columns', ','));
  printf('exo  order: %s\n', strjoin(dynare_exo_columns', ','));
  printf('state_var : %s\n', mat2str(state_var'));
  printf('impact of e_r on r: %.17g\n', impact_r_er);
  printf('written: post82_first_order_goldens.mat + post82_fo_*.csv\n');
  printf('written: post82_kf_data.m, the datafile post82_kf.mod reads\n');
