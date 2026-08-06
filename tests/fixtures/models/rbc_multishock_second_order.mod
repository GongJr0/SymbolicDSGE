// RBC with three correlated shocks, for generating second-order goldens under
// Octave + Dynare -- NOT shipped in the library MODELS/. Run:
//   dynare rbc_multishock_second_order.mod noclearall
//   make_rbc_multishock_second_order_goldens
//
// Goldens: oo_.dr.ghxx, ghxu, ghuu (the shock cross terms the single-shock
// rbc_second_order.mod cannot produce) and ghs2 against a full covariance
// rather than one variance.
//
// Economics is identical to tests/fixtures/models/rbc_multishock_second_order.yaml;
// only the timing convention differs (Dynare predetermined k(-1)/z(-1) vs our
// offset-0/+1 states k(t)/z(t)).
//
// z, d and g each enter through a different nonlinearity, so no cross term is
// zero by construction. All three are zero in steady state, which leaves the
// textbook RBC steady state with gbar taken out of consumption.

var c k z d g;
varexo e_z e_d e_g;
parameters beta gamma alpha delta gbar rho_z rho_d rho_g
           sig_z sig_d sig_g corr_zd corr_zg corr_dg;

beta    = 0.99;
gamma   = 2.0;
alpha   = 0.33;
delta   = 0.025;
gbar    = 0.5;
rho_z   = 0.95;
rho_d   = 0.80;
rho_g   = 0.90;
sig_z   = 0.010;
sig_d   = 0.008;
sig_g   = 0.015;
corr_zd = 0.35;
corr_zg = -0.20;
corr_dg = 0.15;

model;
  c^(-gamma) = beta * exp(d) * c(+1)^(-gamma)
             * (alpha * exp(z(+1)) * k^(alpha - 1) + 1 - delta);
  c + k + gbar * exp(g) = exp(z) * k(-1)^alpha + (1 - delta) * k(-1);
  z = rho_z * z(-1) + e_z;
  d = rho_d * d(-1) + e_d;
  g = rho_g * g(-1) + e_g;
end;

initval;
  z = 0;
  d = 0;
  g = 0;
  k = 28.35;
  c = 1.807;
end;
steady;
check;

shocks;
  var e_z; stderr sig_z;
  var e_d; stderr sig_d;
  var e_g; stderr sig_g;
  corr e_z, e_d = corr_zd;
  corr e_z, e_g = corr_zg;
  corr e_d, e_g = corr_dg;
end;

stoch_simul(order = 2, irf = 0, noprint);
