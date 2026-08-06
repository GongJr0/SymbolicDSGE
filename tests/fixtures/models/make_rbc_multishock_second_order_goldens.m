% make_rbc_multishock_second_order_goldens.m
%   Run after:
%     dynare rbc_multishock_second_order.mod noclearall
%
% Produces the order-2 decision rule for the three-shock RBC: ghx / ghu at first
% order, then ghxx / ghxu / ghuu / ghs2. The point of the model is the blocks
% the single-shock fixture cannot reach, so ghxu and ghuu are the payload here
% and ghxx is the control.
%
% RAW ONLY. Nothing here re-dates, rescales, or re-indexes a Dynare output to
% suit our conventions. Every array is dumped as Dynare produced it, together
% with the index vectors (order_var, state_var) and the column labels, so the
% mapping onto our layout is reproducible from the raw arrays alone and lives on
% our side, in the tests.
%
% Column labels are built here rather than assumed on the Python side. Dynare
% orders the second-order columns as a Kronecker product, so column (i-1)*q + j
% of kron(A, B) pairs A(i) with B(j): ghxx is kron(state, state), ghxu is
% kron(state, exo), ghuu is kron(exo, exo). If that is wrong the labels are
% wrong the same way, and the symmetry of ghxx and ghuu will say so.

  iorder = 2;

  endo_names  = cellstr(M_.endo_names);
  exo_names   = cellstr(M_.exo_names);
  param_names = cellstr(M_.param_names);

  n_endo = M_.endo_nbr;
  n_exo  = M_.exo_nbr;

  ys        = oo_.dr.ys;
  order_var = oo_.dr.order_var(:);
  state_var = oo_.dr.state_var(:);
  Sigma_e   = M_.Sigma_e;

  % Row and column vocabularies. Rows of every gh* array are in DR order; ghx
  % and the second-order state columns follow state_var.
  dr_names    = endo_names(order_var);
  state_names = endo_names(state_var);

  ghx  = oo_.dr.ghx;
  ghu  = oo_.dr.ghu;
  ghxx = oo_.dr.ghxx;
  ghxu = oo_.dr.ghxu;
  ghuu = oo_.dr.ghuu;
  ghs2 = oo_.dr.ghs2;

  n_state = numel(state_names);

  ghxx_cols = cell(1, n_state * n_state);
  for i = 1:n_state
    for j = 1:n_state
      ghxx_cols{(i - 1) * n_state + j} = [state_names{i} '*' state_names{j}];
    end
  end

  ghxu_cols = cell(1, n_state * n_exo);
  for i = 1:n_state
    for j = 1:n_exo
      ghxu_cols{(i - 1) * n_exo + j} = [state_names{i} '*' exo_names{j}];
    end
  end

  ghuu_cols = cell(1, n_exo * n_exo);
  for i = 1:n_exo
    for j = 1:n_exo
      ghuu_cols{(i - 1) * n_exo + j} = [exo_names{i} '*' exo_names{j}];
    end
  end

  % Symmetry is the check that the labels above describe the arrays: swapping
  % the two members of a pair must leave the entry alone.
  sym_err_xx = 0;
  for i = 1:n_state
    for j = 1:n_state
      a = ghxx(:, (i - 1) * n_state + j);
      b = ghxx(:, (j - 1) * n_state + i);
      sym_err_xx = max(sym_err_xx, max(abs(a - b)));
    end
  end
  sym_err_uu = 0;
  for i = 1:n_exo
    for j = 1:n_exo
      a = ghuu(:, (i - 1) * n_exo + j);
      b = ghuu(:, (j - 1) * n_exo + i);
      sym_err_uu = max(sym_err_uu, max(abs(a - b)));
    end
  end

  save('-v7', 'rbc_multishock_second_order_goldens.mat', ...
       'ys', 'order_var', 'state_var', 'Sigma_e', ...
       'ghx', 'ghu', 'ghxx', 'ghxu', 'ghuu', 'ghs2', ...
       'endo_names', 'exo_names', 'dr_names', 'state_names', ...
       'ghxx_cols', 'ghxu_cols', 'ghuu_cols');

  dump = @(name, M) dlmwrite(name, M, 'precision', '%.17g');

  dump('rbc_ms_so_ys.csv',        ys);
  dump('rbc_ms_so_order_var.csv', order_var);
  dump('rbc_ms_so_state_var.csv', state_var);
  dump('rbc_ms_so_Sigma_e.csv',   Sigma_e);
  dump('rbc_ms_so_ghx.csv',       ghx);
  dump('rbc_ms_so_ghu.csv',       ghu);
  dump('rbc_ms_so_ghxx.csv',      ghxx);
  dump('rbc_ms_so_ghxu.csv',      ghxu);
  dump('rbc_ms_so_ghuu.csv',      ghuu);
  dump('rbc_ms_so_ghs2.csv',      ghs2);

% ---------------------------------------------------------------------------
% Paste block. The same arrays as Python literals, one list per DR row, so the
% test file can carry them without a data file and without a reshape.
% ---------------------------------------------------------------------------

  fprintf('\n--- rbc_multishock second order goldens ---\n');
  fprintf('decl  order: %s\n', strjoin(endo_names', ','));
  fprintf('exo   order: %s\n', strjoin(exo_names', ','));
  fprintf('DR    rows : %s\n', strjoin(dr_names', ','));
  fprintf('state cols : %s\n', strjoin(state_names', ','));
  fprintf('ghxx symmetry residual: %.3g\n', sym_err_xx);
  fprintf('ghuu symmetry residual: %.3g\n', sym_err_uu);
  fprintf('\n');

  names = {'GHX', 'GHU', 'GHXX', 'GHXU', 'GHUU', 'GHS2'};
  mats  = {ghx, ghu, ghxx, ghxu, ghuu, ghs2};
  for m = 1:numel(names)
    A = mats{m};
    fprintf('_DYNARE_MS_%s = [\n', names{m});
    for i = 1:size(A, 1)
      fprintf('    [');
      if size(A, 2) > 1
        fprintf('%.17g, ', A(i, 1:end - 1));
      end
      fprintf('%.17g],\n', A(i, end));
    end
    fprintf(']\n\n');
  end

  fprintf('written: rbc_multishock_second_order_goldens.mat + rbc_ms_so_*.csv\n');
