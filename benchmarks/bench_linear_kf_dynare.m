function bench_linear_kf_dynare(workdir, model_file, obs_names, data_file, warmup, reps, output_path)
% Dispatch the common Dynare linear Kalman filter benchmark implementation.
  bench_fixture_linear_kf_dynare(workdir, model_file, obs_names, data_file, warmup, reps, output_path);
end
