// post82_first_order.mod plus the yaml's observation block, so Dynare's own
// smoother runs on it and produces the state paths our filter is checked
// against. Economics are untouched.
//
// Run, in this order:
//   dynare post82_first_order.mod noclearall
//   make_post82_first_order_goldens      // writes post82_kf_data.m
//   dynare post82_kf.mod noclearall nograph
//   make_post82_kf_goldens
//
// calib_smoother displays and saves smoother figures. nograph is a dynare
// command-line option rather than a calib_smoother one, so it cannot be set
// from this file; without it the run leaves eps files under post82_kf/graphs.
//
// OutGap, Infl and Rate are static definitions, so the decision rule for
// g z r x Pi is the same one post82_first_order.mod solves. They are declared
// here rather than there because three extra endogenous variables move the
// ghx / ghu rows the decision-rule goldens are transcribed from.
//
// Measurement error is declared in the shocks block on the observed variables,
// which is how a calibrated model carries an H. sig_me = 1 matches the yaml's
// unit measurement standard deviations and zero measurement correlations.
//
// This model is authored by Ege Güney Kıymaç for SymbolicDSGE and is distributed
// under the MIT license.

var g z r x Pi OutGap Infl Rate;
varexo e_g e_z e_r;
parameters beta kappa tau_inv psi_pi psi_x rho_r rho_g rho_z
           pi_star r_star sig_g sig_z sig_r rho_gz sig_me;

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
sig_me  = 1.00;

model(linear);
  Pi = beta*Pi(+1) + kappa*(x - z);
  x  = x(+1) - tau_inv*(r - Pi(+1)) + g;
  r  = rho_r*r(-1) + (1 - rho_r)*(psi_pi*Pi + psi_x*(x - z)) + e_r;
  g  = rho_g*g(-1) + e_g;
  z  = rho_z*z(-1) + e_z;

  OutGap = x;
  Infl   = 4*Pi + pi_star;
  Rate   = 4*r + (r_star + pi_star);
end;

steady;
check;

shocks;
  var e_g; stderr sig_g;
  var e_z; stderr sig_z;
  var e_r; stderr sig_r;
  corr e_g, e_z = rho_gz;

  var OutGap; stderr sig_me;
  var Infl;   stderr sig_me;
  var Rate;   stderr sig_me;
end;

varobs OutGap Infl Rate;

// lik_init defaults to 1 for a stationary model, the unconditional covariance,
// which is the initialization both libraries read the same way.
calib_smoother(datafile = post82_kf_data, filtered_vars, filter_step_ahead = [1]);
