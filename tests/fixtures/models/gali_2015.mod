/*
 * Adapted from the Dynare replication of Gali (2015),
 * "Monetary Policy, Inflation, and the Business Cycle", Chapter 3.
 *
 * Copyright (C) 2016 Johannes Pfeifer
 * Distributed under GPL-3.0-or-later. See LICENSE.GPL in this directory.
 */

var pi ${\pi}$
    y_gap ${\tilde y}$
    y_nat ${y^{nat}}$
    y ${y}$
    yhat ${\hat y}$
    r_nat ${r^{nat}}$
    r_real ${r^r}$
    i ${i}$
    n ${n}$
    m_real ${m-p}$
    m_growth_ann ${\Delta m}$
    nu ${\nu}$
    a ${a}$
    r_real_ann ${r^{r,ann}}$
    i_ann ${i^{ann}}$
    r_nat_ann ${r^{nat,ann}}$
    pi_ann ${\pi^{ann}}$
    z ${z}$
    c ${c}$
    w_real ${\frac{w}{p}}$
    mu ${\mu}$
    mu_hat ${\hat \mu}$
    ;

varexo eps_a ${\varepsilon_a}$
       eps_nu ${\varepsilon_\nu}$
       eps_z ${\varepsilon_z}$
       ;

parameters alppha ${\alpha}$
           betta ${\beta}$
           rho_a ${\rho_a}$
           rho_nu ${\rho_\nu}$
           rho_z ${\rho_z}$
           siggma ${\sigma}$
           varphi ${\varphi}$
           phi_pi ${\phi_\pi}$
           phi_y ${\phi_y}$
           eta ${\eta}$
           epsilon ${\epsilon}$
           theta ${\theta}$
           ;

siggma = 1;
varphi = 5;
phi_pi = 1.5;
phi_y = 0.125;
theta = 3/4;
rho_nu = 0.5;
rho_z = 0.5;
rho_a = 0.9;
betta = 0.99;
eta = 3.77;
alppha = 1/4;
epsilon = 9;

model(linear);
  #Omega = (1-alppha)/(1-alppha+alppha*epsilon);
  #psi_n_ya = (1+varphi)/(siggma*(1-alppha)+varphi+alppha);
  #lambda = (1-theta)*(1-betta*theta)/theta*Omega;
  #kappa = lambda*(siggma+(varphi+alppha)/(1-alppha));

  pi = betta*pi(+1) + kappa*y_gap;
  y_gap = -1/siggma*(i-pi(+1)-r_nat) + y_gap(+1);
  i = phi_pi*pi + phi_y*yhat + nu;
  r_nat = -siggma*psi_n_ya*(1-rho_a)*a + (1-rho_z)*z;
  r_real = i-pi(+1);
  y_nat = psi_n_ya*a;
  y_gap = y-y_nat;
  nu = rho_nu*nu(-1) + eps_nu;
  a = rho_a*a(-1) + eps_a;
  y = a+(1-alppha)*n;
  z = rho_z*z(-1) - eps_z;
  m_growth_ann = 4*(y-y(-1)-eta*(i-i(-1))+pi);
  m_real = y-eta*i;
  i_ann = 4*i;
  r_real_ann = 4*r_real;
  r_nat_ann = 4*r_nat;
  pi_ann = 4*pi;
  yhat = y-steady_state(y);
  y = c;
  w_real = siggma*c + varphi*n;
  mu = -(siggma+(varphi+alppha)/(1-alppha))*y
       + (1+varphi)/(1-alppha)*a;
  mu_hat = -(siggma+(varphi+alppha)/(1-alppha))*y_gap;
end;

shocks;
  var eps_a = 1^2;
  var eps_nu = 0.25^2;
  var eps_z = 0.5^2;
end;

varobs pi_ann i_ann;

steady;
check;
