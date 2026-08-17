/*
 * Adapted from the Dynare replication of Gali and Monacelli (2005),
 * "Monetary Policy and Exchange Rate Volatility in a Small Open Economy".
 *
 * Copyright (C) 2015 Johannes Pfeifer
 * Distributed under GPL-3.0-or-later. See LICENSE.GPL in this directory.
 */

var pih ${\pi_h}$
    x $x$
    y $y$
    ynat ${\bar y}$
    rnat ${\bar r}$
    r $r$
    s $s$
    pi ${\pi}$
    ystar ${y^*}$
    pistar ${\pi^{*}}$
    n ${n}$
    nx ${nx}$
    real_wage ${w-p}$
    a $a$
    c $c$
    ;

varexo eps_star ${\varepsilon^{*}}$
       eps_a ${\varepsilon^{a}}$;

parameters sigma $\sigma$
           eta $\eta$
           gamma $\gamma$
           phi $\varphi$
           epsilon $\varepsilon$
           theta $\theta$
           beta $\beta$
           alpha $\alpha$
           phi_pi $\phi_\pi$
           rhoa $\rho_a$
           rhoy $\rho_y$
           ;

sigma = 1;
eta = 1;
gamma = 1;
phi = 3;
epsilon = 6;
theta = 0.75;
beta = 0.99;
alpha = 0.4;
phi_pi = 1.5;
rhoa = 0.9;
rhoy = 0.86;

model(linear);
  #omega = sigma*gamma + (1-alpha)*(sigma*eta-1);
  #sigma_a = sigma/((1-alpha)+alpha*omega);
  #Theta = (sigma*gamma-1)+(1-alpha)*(sigma*eta-1);
  #lambda = (1-(beta*theta))*(1-theta)/theta;
  #kappa_a = lambda*(sigma_a+phi);
  #Gamma = (1+phi)/(sigma_a+phi);
  #Psi = -Theta*sigma_a/(sigma_a+phi);

  x = x(+1) - sigma_a^(-1)*(r - pih(+1) - rnat);
  pih = beta*pih(+1) + kappa_a*x;
  rnat = -sigma_a*Gamma*(1-rhoa)*a
         + alpha*sigma_a*(Theta+Psi)*(ystar(+1)-ystar);
  ynat = Gamma*a + alpha*Psi*ystar;
  x = y - ynat;
  y = ystar + sigma_a^(-1)*s;
  pi = pih + alpha*(s-s(-1));
  pistar = 0;
  y = a + n;
  nx = alpha*(omega/sigma-1)*s;
  y = c + alpha*omega/sigma*s;
  real_wage = sigma*c + phi*n;
  a = rhoa*a(-1) + eps_a;
  ystar = rhoy*ystar(-1) + eps_star;
  pih = 0;
end;

shocks;
  var eps_star = 1;
  var eps_a = 1;
end;

varobs pi r;

steady;
check;
