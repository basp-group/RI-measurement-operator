function [z, k] = solver_proj_elipse_fb(v2, r2, y, U, epsilont, z0, max_itr, min_itr, eps)
%%
%A. Onose, A. Dabbech, Y. Wiaux - An accelerated splitting algorithm for radio-interferometric %imaging: when natural and uniform weighting meet, MNRAS 2017, arXiv:1701.01748
%https://github.com/basp-group/SARA-PPD
%%
sc = @(z, radius) z * min(radius/norm(z(:)), 1);
z = z0;
alpha = v2 + r2;
mu = 1/max(U)^2;
zdelta = inf;
k = 0;
while k < min_itr || (k < max_itr && zdelta > eps)
    grad = U .* (z - alpha);
    zo = z;
    z = y + sc(z - mu * grad - y, epsilont);
    zdelta = norm(zo - z)/norm(z);
    k = k + 1;
end

end
