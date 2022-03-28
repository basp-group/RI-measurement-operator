function [G, scale, st] = createG(om, J, N, K, nshift, varargin)
% Create the gridding matrix ``G`` to perform the 2D NUFFT.
%
% Parameters
% ----------
% om : double[M, 2]
%     Normalized coordinates of the target of the data points in the
%     :math:`d`-dimensional Fourier domain (uv frequencies) (``M`` data
%     points).
% J : int[1, 2]
%     Size of the interpolation kernel along each dimension.
% N : int[1, 2]
%     Image size along each dimension.
% K : int[1, 2]
%     Size of the Fourier domain along each dimension.
% nshift : int[1, 2]
%     Phase-shift along each dimension in Fourier space (expressed in
%     number of samples in the Fourier domain).
% ktype : string (varargin)
%     Selected interpolator (among ``'minmax:kb'``, ``'minmax:tuned'`` and
%     ``'kaiser'``, see :mat:func:`nufft.compute_interp_coeffs_extended`
%     for associated kernel documentation). Recommended default is
%     ``'minmax:kb'``.
%
% Returns
% -------
% G : sparse complex[M, prod(K)]
%     Sparse de-gridding interpolation matrix involed in the 2D NUFFT.
% scale : double[:, :] [N]
%     Scaling weights involed in the 2D NUFFT, of size ``N``.
%

% Author: P.-A. Thouvenin (pierreantoine.thouvenin@gmail.com)
%

%%
if ~isempty(varargin)
    ktype = varargin{1};
else
    ktype = 'minmax:kb';
end

% Compute gridding coefficients
% compute_interp_coeffs(om, Nd, Jd, Kd, n_shift, ktype)
st = compute_interp_coeffs(om, N, J, K, nshift, ktype); % st.uu of size [J^2, M]
M = size(om, 1);
J2 = prod(J);

% Generate indices in the sparse G matrix
% if rem(J, 2) > 0 % odd
%    c0 = round(om .* K / (2 * pi)) - (J + 1) / 2; % [M, 2]
% else
%    c0 = floor(om .* K / (2 * pi)) - J / 2;
% end
c0 = floor(om .* K / (2 * pi) - J / 2);

kdy = mod(bsxfun(@plus, (1:J(1)).', c0(:, 1).'), K(1)) + 1; % [J M]
kdx = mod(bsxfun(@plus, (1:J(2)).', c0(:, 2).'), K(2)) + 1; % [J M] row indices of the elements within each area, whose leftmost element row indices are given above
x = reshape(bsxfun(@plus, reshape((kdx - 1) * K(1), [1, J(2), M]), reshape(kdy, [J(1), 1, M])), [J2, M]);

% % Fessler
% for id = 1:2
%     % indices into oversampled FFT components
%     koff = nufft_offset(om(:,id), J(id), K(id)); % [M 1] to leftmost near nbr  % [PA]: computes the matrix indices of the leftmost (first) element in the neighbourhood centered in the given frequency locations ()
%     kd{id} = mod(outer_sum([1:J(id)]', koff'), K(id)) + 1; % [J? M] {1,...,K?} % [PA]: computes the indices of the elements in the region centered around the koff frequencies: just add 1:J, frequencies in {1,...,K} (formula (9)))
%     if id > 1 % trick: pre-convert these indices into offsets!
%         kd{id} = (kd{id}-1) * prod(K(1:(id-1))); % [PA] change reference (indices in the final large matrix G)
%     end
%
% end
% x = kd{1}; % [J1 M] % [PA] values of the indices locations
% for id = 2:2
%     Jprod = prod(J(1:id));
%     x = block_outer_sum(x, kd{id}); % outer sum of indices
%     x = reshape(x, Jprod, M);
% end % now kk and uu are [*Jd M]
% ---

y = bsxfun(@times, ones(J2, 1), 1:M); % [J2 M]
% Create the sparse matrix Gt of size [T, F] -> save for the following
% temporal interpolations
G = sparse(y(:), x(:), st.uu, M, prod(K)); % trim down Gt for application to U (remove unused columns...)
scale = st.sn;

end

% % block_outer_sum()
% %
% % in
% % x1  [J1 M]
% % x2  [J2 M]
% % out
% % y   [J1 J2 M]   y(i1,i2,m) = x1(i1,m) + x2(i2,m)
% %
% function y = block_outer_sum(x1, x2)
% [J1 M] = size(x1);
% [J2 M] = size(x2);
% xx1 = reshape(x1, [J1 1 M]); % [J1 1 M] from [J1 M] % [PA] use bsxfun instead!
% xx1 = xx1(:,ones(J2,1),:); % [J1 J2 M], emulating ndgrid
% xx2 = reshape(x2, [1 J2 M]); % [1 J2 M] from [J2 M]
% xx2 = xx2(ones(J1,1),:,:); % [J1 J2 M], emulating ndgrid
% y = xx1 + xx2; % [J1 J2 M]
% end
