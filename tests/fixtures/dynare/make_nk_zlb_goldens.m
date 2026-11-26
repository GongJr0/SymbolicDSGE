% make_nk_zlb_goldens.m
  % Run after:
  %   dynare nk_zlb_1_constraint.mod noclearall
  %
  % Prints every golden to stdout at %.17g and writes no files. Transcribe the
  % session log into tests/_oracles/dynare_nk_zlb_1_constraint.py.
  %
  % Sections are fenced with '### <name> <shape>' so a block can be located
  % without counting lines. Matrices print one row per line, comma separated.
  %
  % The occbin outputs are read through isfield guards and every block is
  % preceded by its own fieldnames dump: unlike oo_.dr, oo_.occbin has no
  % settled field layout across Dynare versions, so the dumps are what tells us
  % whether a missing block is absent or merely renamed.
  %
  % This is the one-constraint twin of make_rbc_occbin_goldens.m. The regime
  % history carries a single unnumbered regime/regimestart pair rather than two
  % numbered ones, and the linear path is dumped whole rather than clipped: at
  % 30 periods there is nothing to save by clipping it.

  printrow = @(v) fprintf([repmat('%.17g,', 1, numel(v) - 1) '%.17g\n'], v);
  % No trailing newline: several integer runs share one line.
  printint = @(v) fprintf([repmat('%d,', 1, numel(v) - 1) '%d'], v);

  endo_names = cellstr(M_.endo_names);
  exo_names = cellstr(M_.exo_names);
  param_names = cellstr(M_.param_names);

  fprintf('### dynare_version\n%s\n', dynare_version());

  fprintf('### endo_names (%d)\n', numel(endo_names));
  fprintf('%s\n', strjoin(endo_names(:)', ','));
  fprintf('### exo_names (%d)\n', numel(exo_names));
  fprintf('%s\n', strjoin(exo_names(:)', ','));
  fprintf('### param_names (%d)\n', numel(param_names));
  fprintf('%s\n', strjoin(param_names(:)', ','));

  fprintf('### params (%d, in param_names order)\n', numel(M_.params));
  printrow(M_.params(:)');

  fprintf('### steady_state (%d, in endo_names order)\n', numel(oo_.steady_state));
  printrow(oo_.steady_state(:)');

  if isfield(options_, 'occbin') && isfield(options_.occbin, 'simul') ...
      && isfield(options_.occbin.simul, 'periods')
      fprintf('### simul_periods\n%d\n', options_.occbin.simul.periods);
      fprintf('### simul_check_ahead_periods\n%d\n', ...
              options_.occbin.simul.check_ahead_periods);
  end

  % The shock schedule, rebuilt from the .mod's periods/values sugar rather than
  % read back out of Dynare. This is the array our side has to be fed, so it is
  % the one input that must not be inferred from an internal field name. Dynare's
  % own copy is shocks_sequence below; the two disagreeing is the signal that the
  % surprise-shock expansion is not what we assumed.
  %
  % Length is the shock block's own reach, which the .mod pads with zeros out to
  % 30 so that every reported date is also a date Dynare records a regime for.
  T_shock = 30;
  shock_rebuilt = zeros(T_shock, 1);
  shock_rebuilt(6) = 5*0.005;

  fprintf('### shock_rebuilt (%d x 1, epsi)\n', T_shock);
  for i = 1:T_shock
      printrow(shock_rebuilt(i));
  end

  if isfield(oo_, 'occbin')
      occbin_fields = fieldnames(oo_.occbin);
      fprintf('### occbin_fields\n');
      fprintf('%s\n', strjoin(occbin_fields(:)', ','));
  end

  if isfield(oo_, 'occbin') && isfield(oo_.occbin, 'simul')
      simul = oo_.occbin.simul;

      simul_fields = fieldnames(simul);
      fprintf('### occbin_simul_fields\n');
      fprintf('%s\n', strjoin(simul_fields(:)', ','));

      if isfield(simul, 'error_flag')
          fprintf('### error_flag\n%d\n', simul.error_flag);
      end

      if isfield(simul, 'exo_pos')
          fprintf('### exo_pos (index into exo_names)\n');
          printint(simul.exo_pos(:)');
          fprintf('\n');
      end

      % Dynare's own shock array, against which shock_rebuilt above is checked.
      if isfield(simul, 'shocks_sequence')
          fprintf('### shocks_sequence (%d x %d)\n', ...
                  size(simul.shocks_sequence, 1), size(simul.shocks_sequence, 2));
          for i = 1:size(simul.shocks_sequence, 1)
              printrow(simul.shocks_sequence(i, :));
          end
      end

      if isfield(simul, 'ys')
          fprintf('### occbin_ys (%d, in endo_names order)\n', numel(simul.ys));
          printrow(simul.ys(:)');
      end

      % The piecewise path is the whole point. The linear path is the same
      % simulation with the constraint ignored, so it checks our reference
      % solve before any regime logic is involved.
      if isfield(simul, 'piecewise')
          fprintf('### piecewise (%d x %d, columns = endo_names)\n', ...
                  size(simul.piecewise, 1), size(simul.piecewise, 2));
          for i = 1:size(simul.piecewise, 1)
              printrow(simul.piecewise(i, :));
          end
      end

      if isfield(simul, 'linear')
          fprintf('### linear (%d x %d, columns = endo_names)\n', ...
                  size(simul.linear, 1), size(simul.linear, 2));
          for i = 1:size(simul.linear, 1)
              printrow(simul.linear(i, :));
          end
      end

      if isfield(simul, 'regime_history')
          rh = simul.regime_history;
          rh_fields = fieldnames(rh);
          fprintf('### regime_history_fields\n');
          fprintf('%s\n', strjoin(rh_fields(:)', ','));
          fprintf('### regime_history_length\n%d\n', numel(rh));

          % Run-length encoded, one date per line. regime1 holds the 0/1 values
          % and regimestart1 the offsets they take effect at, counted from the
          % current date, so an entry of 1 is the date itself. Entry 1 is
          % therefore the realized regime and the rest is the converged guess
          % over the expectation horizon. That guess, not the realized scalar,
          % is what the backward recursion consumes, which is why it is kept
          % whole.
          %
          % The fields are regime/regimestart, unnumbered: occbin.solver sends a
          % one-constraint model to solve_one_constraint, which writes those,
          % while the numbered regime1/regimestart1 pairs come from
          % solve_two_constraints.
          fprintf('### regime_history (t | regime | regimestart)\n');
          for tt = 1:numel(rh)
              fprintf('%d|', tt);
              printint(rh(tt).regime(:)');
              fprintf('|');
              printint(rh(tt).regimestart(:)');
              fprintf('\n');
          end

          % The realized 0/1 path, which is the fixed point the driver has to
          % reproduce exactly. Derived rather than stored so it cannot drift
          % from the history above.
          fprintf('### realized_regimes (%d x 1, zlb)\n', numel(rh));
          for tt = 1:numel(rh)
              fprintf('%d\n', rh(tt).regime(1));
          end
      end
  end
