function [par, logpost, accepted, neval, sampler_options] = posterior_sampler_iteration(objective_function, last_draw, last_posterior, sampler_options, varargin)
% Benchmark-local Dynare 7.1 dispatcher for Haario adaptive RWMH.

if ~strcmp(sampler_options.posterior_sampling_method, 'random_walk_metropolis_hastings')
  error('bench_mcmc_dynare only supports random_walk_metropolis_hastings.');
end

[par, logpost, accepted, sampler_options] = random_walk_metropolis_hastings( ...
    objective_function, last_draw, last_posterior, sampler_options, varargin{:});
neval = 1;
end
