function bench_mcmc_dynare(workdir, model_name, draws, burn_in, warmup, reps, seed, adapt, adapt_start, shared_mode_path, output_path)
  previous_dir = pwd;
  restore_dir = onCleanup(@() cd(previous_dir));
  cd(workdir);
  dynare(model_name, 'noclearall', 'nograph');

  addpath(fileparts(mfilename('fullpath')), '-begin');
  sampler_path = which('posterior_sampler_iteration');
  expected_sampler_path = fullfile(fileparts(mfilename('fullpath')), 'posterior_sampler_iteration.m');
  if ~strcmpi(sampler_path, expected_sampler_path)
    error('The benchmark-local Haario sampler dispatcher is not active.');
  end

  global M_ options_ oo_ estim_params_ bayestopt_ dataset_ dataset_info
  apply_shared_mode(M_, shared_mode_path);
  M0 = M_; options0 = options_; oo0 = oo_; estim_params0 = estim_params_;
  bayestopt0 = bayestopt_; dataset0 = dataset_; dataset_info0 = dataset_info;
  for i = 1:warmup
    run_chain(M0, options0, oo0, estim_params0, bayestopt0, dataset0, dataset_info0, draws, burn_in, seed + i - 1, adapt, adapt_start);
  end

  times = zeros(reps, 1);
  samples = [];
  accept_rate = NaN;
  for i = 1:reps
    started = tic;
    [samples, accept_rate] = run_chain(M0, options0, oo0, estim_params0, bayestopt0, dataset0, dataset_info0, draws, burn_in, seed + warmup + i - 1, adapt, adapt_start);
    times(i) = toc(started);
  end
  save('-v7', output_path, 'times', 'samples', 'accept_rate');
end

function apply_shared_mode(M_, shared_mode_path)
  % Start every chain from the MAP the caller found, not from the one this
  % setup pass found, and propose from the covariance found there. Both are
  % passed in, so no Hessian is built on this side: dynare_estimation_1 stores
  % a Hessian and inverts it into the jumping covariance, so the mode file
  % takes the inverse of the covariance the native side samples with. The
  % setup pass ran under cova_compute = 0 and left no hh of its own.
  mode_path = fullfile(M_.dname, 'Output', [M_.fname '_mode.mat']);
  target = load(mode_path);
  shared = load(shared_mode_path);
  names = cellstr(target.parameter_names);
  shared_names = cellstr(shared.parameter_names);
  if numel(names) ~= numel(shared_names)
    error('The shared mode holds %d parameters, the mode file %d.', ...
          numel(shared_names), numel(names));
  end
  order = zeros(numel(names), 1);
  for i = 1:numel(names)
    j = find(strcmp(shared_names, strtrim(names{i})), 1);
    if isempty(j)
      error('The shared mode has no entry for estimated parameter ''%s''.', names{i});
    end
    order(i) = j;
  end
  xparam1 = shared.xparam1(order);
  hh = inv(shared.proposal_cov(order, order));
  parameter_names = target.parameter_names;
  fval = target.fval;
  save('-v7', mode_path, 'xparam1', 'hh', 'parameter_names', 'fval');
end

function [samples, accept_rate] = run_chain(M0, options0, oo0, estim_params0, bayestopt0, dataset0, dataset_info0, draws, burn_in, seed, adapt, adapt_start)
  global M_ options_ oo_ estim_params_ bayestopt_ dataset_ dataset_info amh_t0
  M_ = M0; options_ = options0; oo_ = oo0; estim_params_ = estim_params0;
  bayestopt_ = bayestopt0; dataset_ = dataset0; dataset_info = dataset_info0;
  total_steps = draws + burn_in;
  options_.mode_compute = 0;
  options_.mode_file = fullfile(M_.dname, 'Output', [M_.fname '_mode.mat']);
  % The mode file carries the proposal, so this reads hh rather than building one.
  options_.cova_compute = 1;
  options_.mh_replic = total_steps;
  options_.mh_nblck = 1;
  options_.mh_drop = burn_in / total_steps;
  options_.nodiagnostic = true;
  options_.mh_tune_jscale.status = false;
  options_.posterior_sampler_options.posterior_sampling_method = 'random_walk_metropolis_hastings';
  if adapt
    amh_t0 = adapt_start;
  else
    amh_t0 = Inf;
  end
  rng(seed, 'twister');
  metropolis_dir = fullfile(M_.dname, 'metropolis');
  if exist(metropolis_dir, 'dir')
    rmdir(metropolis_dir, 's');
  end
  dynare_estimation({}, M_.dname);
  samples = GetAllPosteriorDraws(options_, M_.dname, M_.fname, 'all');
  record = load_last_mh_history_file(metropolis_dir, M_.fname);
  accept_rate = record.AcceptanceRatio(1);
end
