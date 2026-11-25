% make_post82_kf_goldens.m
%   Run after:
%     dynare post82_first_order.mod noclearall
%     make_post82_first_order_goldens        % writes post82_kf_data.m
%     dynare post82_kf.mod noclearall nograph
%
% Dumps the state paths Dynare's own smoother produced, for the three variables
% our filter is compared on plus the two it solves for.
%
% INDEXING. Dynare dates a prediction by the information set behind it rather
% than by the period being estimated. From the manual's filtered_vars entry and
% from store_smoother_results.m:
%
%   oo_.UpdatedVariables   a_{t|t}     dated by the period estimated
%   oo_.FilteredVariables  a_{t+1|t}   dated by the information set, so the
%                                      array runs to one period past the sample
%   oo_.SmoothedVariables  a_{t|T}     dated by the period estimated
%
% Our x_pred uses the other rule, so it holds the same predictions one row
% earlier. Nothing is shifted here. Reindexing is our side's problem and belongs
% in our tests, which is the same rule the decision-rule goldens follow.
%
% RAW ONLY, with one exception that is not a choice: Dynare adds the steady
% state back before storing, and ys is zero for every variable below, so the
% arrays are deviations either way.

  decl = {'g', 'z', 'r', 'x', 'Pi'};
  n_decl = numel(decl);

  assert(isfield(oo_, 'UpdatedVariables'), ...
         'no smoother results: did dynare post82_kf.mod run?');
  assert(isfield(oo_, 'FilteredVariables'), ...
         'no filtered variables: calib_smoother needs the filtered_vars option');

  ys_decl = zeros(n_decl, 1);
  endo_names = cellstr(M_.endo_names);
  for jj = 1:n_decl
    ys_decl(jj) = oo_.dr.ys(find(strcmp(endo_names, decl{jj}), 1));
  end
  assert(max(abs(ys_decl)) < 1e-12, ...
         'a compared variable has a nonzero steady state; the dumps are not deviations');

  T_upd  = numel(oo_.UpdatedVariables.(decl{1}));
  T_filt = numel(oo_.FilteredVariables.(decl{1}));

  kf_updated  = zeros(T_upd,  n_decl);
  kf_smoothed = zeros(T_upd,  n_decl);
  kf_filtered = zeros(T_filt, n_decl);

  for jj = 1:n_decl
    kf_updated(:, jj)  = oo_.UpdatedVariables.(decl{jj})(:);
    kf_smoothed(:, jj) = oo_.SmoothedVariables.(decl{jj})(:);
    kf_filtered(:, jj) = oo_.FilteredVariables.(decl{jj})(:);
  end

  kf_decl_columns = decl;

% ---------------------------------------------------------------------------
% SAVE
% ---------------------------------------------------------------------------

  save('-v7', 'post82_kf_goldens.mat', ...
       'kf_updated', 'kf_filtered', 'kf_smoothed', 'kf_decl_columns');

  dump = @(name, M) dlmwrite(name, M, 'precision', '%.17g');

  dump('post82_kf_updated.csv',  kf_updated);
  dump('post82_kf_filtered.csv', kf_filtered);
  dump('post82_kf_smoothed.csv', kf_smoothed);

  printf('\n--- post82 kalman goldens, dynare smoother ---\n');
  printf('columns  : %s\n', strjoin(kf_decl_columns, ','));
  printf('updated  : %d rows (a_{t|t})\n',   size(kf_updated, 1));
  printf('filtered : %d rows (a_{t+1|t})\n', size(kf_filtered, 1));
  printf('smoothed : %d rows (a_{t|T})\n',   size(kf_smoothed, 1));
  printf('written: post82_kf_goldens.mat + post82_kf_*.csv\n');
