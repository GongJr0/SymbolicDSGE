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

shocks;
  var e_g; stderr sig_g;
  var e_z; stderr sig_z;
  var e_r; stderr sig_r;
  corr e_g, e_z = rho_gz;
end;

varobs OutGap Infl Rate;

steady;
check;
