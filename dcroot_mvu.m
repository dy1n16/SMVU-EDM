function x = dcroot_mvu(omega, alpha, pidx, zidx)
% To solve
% min 0.5*(x - omega)^2 - 2*alpha*x^(0.5) s.t. x>=0. (where alpha>=0)
% In our application, both omega and alpha are matrices
% pidx = alpha > 0. Since it does not change, we
% zidx = alpha = 0; both pidx and zidx are for upper part of alpha
% NOTE: when computing the cubic root, nthroot should be used to avoid
% complex solutions, which would happen when a^(1/3) is used.
% diagonal of x is forced to be 0, tailed to EDM optimization

% set the size of the problem
 [m, n] = size(omega);
 x = zeros(m, n);

% for alpha = 0 part, x = max(0, omega)
 x(zidx) = max(0, omega(zidx));

% for alpha > 0 part, compute x
 a = alpha(pidx);
 w = omega(pidx);
 u = a/2; 
 v = w/3;
 tau = u.^2 - v.^3;
 
 ell = length(a);
 y = zeros(ell, 1);
 
 IndTau = tau < 0;
 tauphi = u(IndTau) ./ sqrt( v(IndTau).^3) ;

 y(IndTau) = 4*v(IndTau) .* ( cos( acos(tauphi)/3 ) ).^2;
 y(~IndTau) = ( nthroot( u(~IndTau) + sqrt(tau(~IndTau)), 3) + ...
              nthroot( u(~IndTau) - sqrt(tau(~IndTau)), 3) ).^2;
% set x for the LIndex part
 x(pidx) = y; 
 x = x + x'; % to make x symmetric

end

