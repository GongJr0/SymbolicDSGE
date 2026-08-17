/*
 * Adapted from the Dynare replication of Ireland (2004),
 * "Technology Shocks in the New Keynesian Model".
 *
 * Copyright (C) 2016 Johannes Pfeifer
 * Distributed under GPL-3.0-or-later. See LICENSE.GPL in this directory.
 */

var a ${a}$
    e ${e}$
    z ${az}$
    x ${x}$
    pihat ${\hat p}$
    yhat ${\hat y}$
    ghat ${\hat g}$
    rhat ${\hat r}$
    gobs ${g^{obs}}$
    robs ${r^{obs}}$
    piobs ${\pi^{obs}}$
    r_annual ${r^{ann}}$
    pi_annual ${\pi^{ann}}$
    ;

varexo eps_a ${\varepsilon_a}$
       eps_e ${\varepsilon_e}$
       eps_z ${\varepsilon_z}$
       eps_r ${\varepsilon_r}$
       ;

parameters beta ${\beta}$
           alpha_x ${\alpha}$
           alpha_pi ${\alpha_\pi}$
           rho_a ${\rho_a}$
           rho_e ${\rho_e}$
           omega ${\omega}$
           psi ${\psi}$
           rho_pi ${\rho_\pi}$
           rho_g ${\rho_g}$
           rho_x ${\rho_x}$
           ;

beta = 0.99;
psi = 0.1;
omega = 0.0581;
alpha_x = 0.00001;
alpha_pi = 0.00001;
rho_pi = 0.3866;
rho_g = 0.3960;
rho_x = 0.1654;
rho_a = 0.9048;
rho_e = 0.9907;

model(linear);
  a = rho_a*a(-1) + eps_a;
  e = rho_e*e(-1) + eps_e;
  z = eps_z;
  x = alpha_x*x(-1) + (1-alpha_x)*x(+1) - (rhat-pihat(+1))
      + (1-omega)*(1-rho_a)*a;
  pihat = beta*(alpha_pi*pihat(-1)+(1-alpha_pi)*pihat(+1)) + psi*x - e;
  x = yhat - omega*a;
  ghat = yhat-yhat(-1) + z;
  rhat-rhat(-1) = rho_pi*pihat + rho_g*ghat + rho_x*x + eps_r;
  gobs = ghat;
  robs = rhat;
  piobs = pihat;
  r_annual = 4*rhat;
  pi_annual = 4*pihat;
end;

shocks;
  var eps_a; stderr 0.0302;
  var eps_e; stderr 0.0002;
  var eps_z; stderr 0.0089;
  var eps_r; stderr 0.0028;
end;

varobs gobs piobs;

steady;
check;
