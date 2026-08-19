function [par, logpost, accepted, sampler_options] = random_walk_metropolis_hastings(objective_function, last_draw, last_posterior, sampler_options, varargin)
% Benchmark-local Haario adaptive random-walk Metropolis-Hastings step.
%
% This adapts the original whole-chain implementation to Dynare 7.1's
% posterior_sampler_iteration interface. It preserves its Haario covariance
% recursion and global amh_t0 adaptation-start control.
%
% Per the notice below, this file is distributed under the GNU General Public License (GPL) version 3 or later.
% The full license text can be found at https://www.gnu.org/licenses/gpl-3.0.html and in this repository
% (./tests/fixtures/models/LICENSE.GPL) alongside Dynare model fixtures.


% Copyright (C) 2006-2008 Dynare Team
%
% This file is part of Dynare.
%
% Dynare is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.


global amh_t0;
if isempty(amh_t0)
  amh_t0 = Inf;
end

ProposalFun = sampler_options.proposal_distribution;
proposal_factor = sampler_options.proposal_covariance_Cholesky_decomposition;
n = sampler_options.n;
npar = numel(last_draw);

if ~isfield(sampler_options, 'amh_iteration')
  sampler_options.amh_iteration = 0;
  sampler_options.amh_mean = last_draw;
  sampler_options.amh_sum = zeros(size(last_draw));
  bayestopt = varargin{6};
  sampler_options.amh_jscale = diag(bayestopt.jscale(:));
end

par = feval(ProposalFun, last_draw, proposal_factor, n);
mh_bounds = sampler_options.bounds;
if all(par(:) > mh_bounds.lb) && all(par(:) < mh_bounds.ub)
  try
    logpost = -feval(objective_function, par(:), varargin{:});
  catch
    logpost = -Inf;
  end
else
  logpost = -Inf;
end

if isfinite(logpost) && log(rand) < logpost-last_posterior
  accepted = 1;
else
  accepted = 0;
  par = last_draw;
  logpost = last_posterior;
end

sampler_options.amh_iteration = sampler_options.amh_iteration + 1;
j = sampler_options.amh_iteration;

% Haario et al. (2001) recursion, as in the supplied implementation. The
% sampler state replaces the former x2 history buffer in Dynare 7.1.
if isfinite(amh_t0) && j <= amh_t0
  sampler_options.amh_sum = sampler_options.amh_sum + par;
end
if j == amh_t0
  sampler_options.amh_mean = sampler_options.amh_sum/j;
elseif j > amh_t0
  mean0 = sampler_options.amh_mean;
  mean1 = (mean0*(j-1) + par)/j;
  proposal_scale = sampler_options.amh_jscale;
  unscaled_factor = proposal_factor / proposal_scale;
  covariance0 = unscaled_factor' * unscaled_factor;
  covariance1 = (j-1)*covariance0/j + ...
      (j*mean0'*mean0 - (j+1)*mean1'*mean1 + par'*par + 1e-8*eye(npar))/j;
  [new_factor, status] = chol(covariance1);
  if status == 0
    sampler_options.proposal_covariance_Cholesky_decomposition = new_factor * proposal_scale;
  end
  sampler_options.amh_mean = mean1;
end
end
