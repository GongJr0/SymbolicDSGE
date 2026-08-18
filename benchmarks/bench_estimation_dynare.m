function bench_estimation_dynare(workdir, model_name, probes, include_prior, warmup, reps, output_path)
  previous_dir = pwd;
  restore_dir = onCleanup(@() cd(previous_dir));
  cd(workdir);
  dynare(model_name, 'noclearall', 'nograph');

  global M_ options_ oo_ estim_params_ bayestopt_ dataset_ dataset_info
  M0 = M_; options0 = options_; oo0 = oo_; estim_params0 = estim_params_;
  bayestopt0 = bayestopt_; dataset0 = dataset_; dataset_info0 = dataset_info;
  probe_target = zeros(1, size(probes, 2));
  for i = 1:size(probes, 2)
    M_ = M0; options_ = options0; oo_ = oo0; estim_params_ = estim_params0;
    bayestopt_ = bayestopt0; dataset_ = dataset0; dataset_info = dataset_info0;
    probe_target(i) = benchmark_target(probes(:, i), M_, estim_params_, oo_, options_, bayestopt_, include_prior);
  end
  for i = 1:warmup
    for j = 1:size(probes, 2)
      M_ = M0; options_ = options0; oo_ = oo0; estim_params_ = estim_params0;
      bayestopt_ = bayestopt0; dataset_ = dataset0; dataset_info = dataset_info0;
      benchmark_target(probes(:, j), M_, estim_params_, oo_, options_, bayestopt_, include_prior);
    end
  end
  objective_times = zeros(reps, 1);
  for i = 1:reps
    started = tic;
    for j = 1:size(probes, 2)
      M_ = M0; options_ = options0; oo_ = oo0; estim_params_ = estim_params0;
      bayestopt_ = bayestopt0; dataset_ = dataset0; dataset_info = dataset_info0;
      benchmark_target(probes(:, j), M_, estim_params_, oo_, options_, bayestopt_, include_prior);
    end
    objective_times(i) = toc(started) / size(probes, 2);
  end

  for i = 1:warmup
    M_ = M0; options_ = options0; oo_ = oo0; estim_params_ = estim_params0;
    bayestopt_ = bayestopt0; dataset_ = dataset0; dataset_info = dataset_info0;
    dynare_estimation({}, M_.dname);
  end
  times = zeros(reps, 1);
  theta = zeros(size(probes, 1), 1);
  for i = 1:reps
    M_ = M0; options_ = options0; oo_ = oo0; estim_params_ = estim_params0;
    bayestopt_ = bayestopt0; dataset_ = dataset0; dataset_info = dataset_info0;
    started = tic;
    dynare_estimation({}, M_.dname);
    times(i) = toc(started);
    theta = oo_.posterior.optimization.mode(:);
  end
  M_ = M0; options_ = options0; oo_ = oo0; estim_params_ = estim_params0;
  bayestopt_ = bayestopt0; dataset_ = dataset0; dataset_info = dataset_info0;
  terminal_target = benchmark_target(theta, M_, estim_params_, oo_, options_, bayestopt_, include_prior);
  measurement_covariance = M_.H;
  save('-v7', output_path, 'times', 'objective_times', 'theta', 'terminal_target', 'probe_target', 'measurement_covariance');
end

function target = benchmark_target(theta, M, estim_params, oo, options, bayestopt, include_prior)
  logprior = evaluate_prior(theta, M, estim_params, oo, options, bayestopt);
  loglik = evaluate_likelihood(theta, M, estim_params, oo, options, bayestopt);
  target = loglik + include_prior*logprior;
end
