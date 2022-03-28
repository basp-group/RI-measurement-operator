function [z, k] = solver_proj_elipse_fb(v2, r2, y, U, epsilont, z0, max_itr, min_itr, eps)
% Compute the projection of `v2 + r2` onto the :math:`\ell_2` ball
% centered in `y`, of radius `epsilont` with respect to the norm
% induced by the diagonal preconditioner `U` with the forward-backward
% algorithm.
%
% Parameters
% ----------
% v2 : complex[:]
%     Part of the input point to be projected onto the ellipsoid.
% r2 : complex[:]
%     Part of the input point to be projected onto the ellipsoid.
% y : complex[:]
%     Visibility vector, center of the :math:`\ell_2` ball.
% U : double[:]
%     Diagonal preconditioning matrix (encoded as a vector).
% epsilont : double
%     Radius of the :math:`\ell_2` ball.
% z0 : complex[:]
%     Algorithm starting point.
% max_itr : int
%     Maximum number of iterations.
% min_itr : int
%     Minimum number of iterations.
% eps : double
%     Stopping criterion tolerance (relative variation in norm between two
%     consecutive iterates).
%
% Returns
% -------
% z : complex[:]
%     Projection onto the ellipsoid.
% k : int
%     Number of iterations required to converge.
%

%%
sc = @(z, radius) z * min(radius / norm(z(:)), 1);
z = z0;
alpha = v2 + r2;
mu = 1 / max(U)^2;
zdelta = inf;
k = 0;
while k < min_itr || (k < max_itr && zdelta > eps)
    grad = U .* (z - alpha);
    zo = z;
    z = y + sc(z - mu * grad - y, epsilont);
    zdelta = norm(zo - z) / norm(z);
    k = k + 1;
end

end
