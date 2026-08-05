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
% 3. KALMAN FILTER
% ---------------------------------------------------------------------------
% State space (declaration order, deviations from ys):
%     x_t = A_decl x_{t-1} + R_decl e_t,   e_t ~ N(0, Sigma_e)
%     y_t = d + Z x_t + w_t,               w_t ~ N(0, H)
%
% Measurement block is the yaml's observables section:
%     OutGap = x
%     Infl   = 4*Pi + pi_star
%     Rate   = 4*r + (r_star + pi_star)
% with unit measurement standard deviations and zero measurement correlations,
% so H is the identity.
%
% TIMING UNDER TEST: the observation at t pairs with the state at t. A one-period
% slip moves the loglik by far more than any tolerance, which is why this golden
% settles dating in a way no matrix comparison does. The per-period densities are
% dumped alongside the total so a slip is located rather than merely detected.

  Z = zeros(3, n_endo);
  Z(1, idx_x)  = 1.0;
  Z(2, idx_Pi) = 4.0;
  Z(3, idx_r)  = 4.0;
  d = [0.0; pi_star; r_star + pi_star];
  H = eye(3);

  % Data are the observables implied by the stochastic path above, dumped so
  % both sides filter identical numbers rather than regenerating a path. One row
  % per simulated period.
  T_kf    = size(stoch_dynare, 1);
  kf_data = (d * ones(1, T_kf) + Z * stoch_dynare')';   % (T_kf, 3)

  % --- P0, both defensible choices ---------------------------------------
  % (a) unconditional covariance by doubling, no control-package dependency
  Q  = Sigma_e;
  RQ = R_decl * Q * R_decl';
  P_uncond = RQ;
  Ak = A_decl;
  for iter = 1:60
    P_uncond = P_uncond + Ak * P_uncond * Ak';
    Ak = Ak * Ak;
  end
  P_uncond = 0.5 * (P_uncond + P_uncond');   % symmetrize drift

  % (b) the yaml's own P0: diag 1.0 on every variable. The two libraries do not
  % mean the same thing by that. kalman_filter.m documents its P argument as
  % Var_0(alpha_1), the covariance of the state the FIRST observation updates.
  % Ours is the covariance one period earlier, and the filter predicts before it
  % updates, so ours reaches the first update at A*P0*A' + RQR. Both readings are
  % dumped. They agree under P_uncond, which is a fixed point of that map, and
  % nowhere else, so a test that only ever uses P_uncond cannot see the gap.
  P_fixed_dynare = eye(n_endo);
  P_fixed_ours   = A_decl * eye(n_endo) * A_decl' + RQ;
  % ------------------------------------------------------------------------

  % Dynare's filter takes the data with the constant already removed and one
  % column per period. Zflag = 1 passes Z as a matrix rather than an index
  % vector, which is what lets a measurement block with loadings other than 1
  % go through at all.
  Y_kf = (kf_data - ones(T_kf, 1) * d')';

  % riccati_tol = 0 keeps the full recursion for every period. At Dynare's
  % default the gain freezes once it stops moving, which is a fine thing for an
  % estimation loop to do and the wrong thing for a golden.
  run_dynare_kf = @(P0) kalman_filter( ...
      Y_kf, 1, T_kf, zeros(n_endo, 1), P0, ...
      1e-10, 0, false, 0, ...
      A_decl, Q, R_decl, H, Z, ...
      n_endo, size(Z, 1), n_exo, 1, 0, {}, false, false);

  % kalman_filter returns MINUS the log-likelihood, 2*pi constant already in,
  % and LIKK holds the same quantity period by period.
  [neg_uncond,       likk_uncond]       = run_dynare_kf(P_uncond);
  [neg_fixed_dynare, likk_fixed_dynare] = run_dynare_kf(P_fixed_dynare);
  [neg_fixed_ours,   likk_fixed_ours]   = run_dynare_kf(P_fixed_ours);

  kf_loglik_uncond       = -neg_uncond;
  kf_loglik_fixed_dynare = -neg_fixed_dynare;
  kf_loglik_fixed_ours   = -neg_fixed_ours;

  kf_likk_uncond       = -likk_uncond(:)';
  kf_likk_fixed_dynare = -likk_fixed_dynare(:)';
  kf_likk_fixed_ours   = -likk_fixed_ours(:)';

  assert(abs(sum(kf_likk_uncond) - kf_loglik_uncond) < 1e-10, ...
         'per-period densities do not sum to the total');

  % Datafile for post82_kf.mod. Column vectors named for the varobs block there.
  obs_names = {'OutGap', 'Infl', 'Rate'};
  fid = fopen('post82_kf_data.m', 'w');
  fprintf(fid, '%% Written by make_post82_first_order_goldens.m. Do not edit.\n');
  for jj = 1:numel(obs_names)
    fprintf(fid, '%s = [\n', obs_names{jj});
    fprintf(fid, '%.17g\n', kf_data(:, jj));
    fprintf(fid, '];\n');
  end
  fclose(fid);

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
       'kf_data', 'P_uncond', 'P_fixed_dynare', 'P_fixed_ours', 'Z', 'd', 'H', ...
       'kf_loglik_uncond', 'kf_loglik_fixed_dynare', 'kf_loglik_fixed_ours', ...
       'kf_likk_uncond', 'kf_likk_fixed_dynare', 'kf_likk_fixed_ours', ...
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
  dump('post82_fo_kf_data.csv',   kf_data);
  dump('post82_fo_kf_Z.csv',      Z);
  dump('post82_fo_kf_d.csv',      d);
  dump('post82_fo_kf_H.csv',      H);
  dump('post82_fo_kf_P_uncond.csv',      P_uncond);
  dump('post82_fo_kf_P_fixed_ours.csv',  P_fixed_ours);
  dump('post82_fo_kf_likk.csv', [kf_likk_uncond; kf_likk_fixed_dynare; kf_likk_fixed_ours]);
  dump('post82_fo_kf_loglik.csv', [kf_loglik_uncond, kf_loglik_fixed_dynare, ...
                                   kf_loglik_fixed_ours]);

  printf('\n--- post82 first order goldens ---\n');
  printf('decl order: %s\n', strjoin(dynare_decl_columns', ','));
  printf('exo  order: %s\n', strjoin(dynare_exo_columns', ','));
  printf('state_var : %s\n', mat2str(state_var'));
  printf('impact of e_r on r: %.17g\n', impact_r_er);
  printf('loglik  P0 uncond      : %.17g\n', kf_loglik_uncond);
  printf('loglik  P0 = I, dynare : %.17g\n', kf_loglik_fixed_dynare);
  printf('loglik  P0 = I, ours   : %.17g\n', kf_loglik_fixed_ours);
  printf('written: post82_first_order_goldens.mat + post82_fo_*.csv\n');
  printf('written: post82_kf_data.m, the datafile post82_kf.mod reads\n');
