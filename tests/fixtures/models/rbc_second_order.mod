// Canonical RBC (CRRA utility, Cobb-Douglas production, one TFP shock) for
// generating second-order goldens under Octave + Dynare -- NOT shipped in the
// library MODELS/. Run:  dynare rbc_second_order.mod
// Goldens: oo_.dr.ghxx (g_xx) and oo_.dr.ghs2 (the sigma^2 risk correction),
// in Dynare's decision-rule ordering (map to our layout when comparing).
//
// Economics is identical to tests/fixtures/models/rbc_second_order.yaml; only
// the timing convention differs (Dynare predetermined k(-1)/z(-1) vs our
// offset-0/+1 states k(t)/z(t)).

var c k z;
varexo e;
parameters beta gamma alpha delta rho sig;

beta  = 0.99;
gamma = 2.0;
alpha = 0.33;
delta = 0.025;
rho   = 0.95;
sig   = 0.01;

model;
  c^(-gamma) = beta * c(+1)^(-gamma)
             * (alpha * exp(z(+1)) * k^(alpha - 1) + 1 - delta);
  c + k = exp(z) * k(-1)^alpha + (1 - delta) * k(-1);
  z = rho * z(-1) + e;
end;

initval;
  z = 0;
  k = 28.351;
  c = 2.3068;
end;
steady;
check;

shocks;
  var e;
  stderr sig;
end;

stoch_simul(order = 2, irf = 0, noprint);
