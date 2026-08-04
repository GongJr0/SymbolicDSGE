// New Keynesian model, economics identical to tests/fixtures/models/POST82.yaml
// and MODELS/POST82.yaml. Order-1 goldens generator, NOT shipped in MODELS/.
// Run:  dynare post82_first_order.mod noclearall
// then: make_post82_first_order_goldens
//
// Written at Dynare's own dating throughout: every equation sits at t, lags are
// explicit, and innovations are dated t. The yaml is written the same way, so
// nothing here is a re-spelling of the model to suit our solver.
//
// This model is the reason the goldens are worth generating. e_r enters the
// Taylor rule, which carries Pi(t) and x(t), so the response of r to its own
// innovation is a fixed point rather than one. See issue #390.
//
// pi_star and r_star do not appear in the model block. They are the measurement
// intercepts for Infl and Rate and are declared so the golden script can read
// them off M_.params rather than repeating the calibration.

var g z r x Pi;
varexo e_g e_z e_r;
parameters beta kappa tau_inv psi_pi psi_x rho_r rho_g rho_z
           pi_star r_star sig_g sig_z sig_r rho_gz;

beta    = 0.971;
kappa   = 0.58;
tau_inv = 1.86;
psi_pi  = 2.19;
psi_x   = 0.30;
rho_r   = 0.84;
rho_g   = 0.83;
rho_z   = 0.85;
pi_star = 3.43;
r_star  = 3.01;
sig_g   = 0.18;
sig_z   = 0.64;
sig_r   = 0.18;
rho_gz  = 0.36;

model(linear);
  Pi = beta*Pi(+1) + kappa*(x - z);
  x  = x(+1) - tau_inv*(r - Pi(+1)) + g;
  r  = rho_r*r(-1) + (1 - rho_r)*(psi_pi*Pi + psi_x*(x - z)) + e_r;
  g  = rho_g*g(-1) + e_g;
  z  = rho_z*z(-1) + e_z;
end;

steady;
check;

shocks;
  var e_g; stderr sig_g;
  var e_z; stderr sig_z;
  var e_r; stderr sig_r;
  corr e_g, e_z = rho_gz;
end;

stoch_simul(order = 1, irf = 0, noprint);
