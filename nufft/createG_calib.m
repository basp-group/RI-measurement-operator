function [G, scale, ll, v, V] = createG_calib(D, om, N, K, J, S, nshift, varargin)
% Create the gridding matrix ``G`` to perform the 2D NUFFT including DDE
% kernels.
%
% Parameters
% ----------
% D  : double[:, :]
%     DDEs kernels for a single time instant ``[S2, na]``.
% om : double[M, 2]
%     Normalized coordinates of the target of the data points in the
%     :math:`d`-dimensional Fourier domain (uv frequencies).
% N : int[1, 2]
%     Image size along each dimension.
% K : int[1, 2]
%     Size of the Fourier domain along each dimension.
% J : int
%     Size of the square interpolation kernel along a single dimension.
% S : int
%     Size of the square DDE kernels (in the spatial Fourier domain) along
%     a single dimension. In the following, ``S2 = S^2``.
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
% ll : int[:]
%     Position of the nonzero values in each row of ``G``.
% v : double[Q2, M]
%      convolutions between the DDE kernels (``Q2 = (2 * S - 1)^2``).
% V : double[Q2, J2, M]
%      values contained in ``G`` (``J2 = J^2``).
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
% compute_interp_coeffs(om, Nd, Jd, Kd, n_shift)
st = compute_interp_coeffs(om, N, [J, J], K, nshift, ktype); % st.uu of size [J^2, M]
Q  = 2 * S - 1;             % size of the kernels after convolution
Q2 = Q^2;
J2 = J^2;

Qprime = floor(Q / 2);      % Q is always odd (one possibility only)
tmp1 = (-Qprime:Qprime).';
tmp1 = tmp1(:, ones(Q, 1));

[~, na, ~] = size(D);    % [S2, na] number of antennas at time t
M_true  = na * (na - 1) / 2; % number of acquisitions -> check value...
M = size(om, 1);       % M = M_true if all the measurements are present, M < M_true if some of the data have been flagged: need to add flagging in this case
T = size(D, 3);
v = zeros(Q^2, M); % convolution values (stored in column for each pair)

%% Perform 2D convolutions and gridding using D1 and D2 (to be possibly performed in parallel)
for t = 1:T
    q = 0; % global counter
    for alpha = 1:na - 1
        for beta = alpha + 1:na % modify the double loop to exclusively select the appropriate elements, apply nonzeros on W
            % 2D convolutions
            q = q + 1;
            v(:, q) = reshape(conv2(rot90(reshape(D(:, alpha, t), [S, S]), 2), reshape(conj(D(:, beta, t)), [S, S])), [Q^2, 1]); % only select the appropriate entries...
        end
    end
end

% % shortcut for testing purposes
% vt = reshape(conv2(rot90(reshape(D(:,1,1),[S,S]),2),reshape(conj(D(:,2,1)),[S,S])), [Q^2,1]);
% v = vt(:, ones(M, 1));

% Generate indices in the sparse G matrix
if rem(J, 2) > 0 % odd
   c0 = round(om .* K / (2 * pi)) - (J + 1) / 2; % [M, 2]
else
   c0 = floor(om .* K / (2 * pi)) - J / 2;
end
kdy = bsxfun(@plus, (1:J).', c0(:, 1).'); % [J M]
kdx = bsxfun(@plus, (1:J).', c0(:, 2).'); % [J M]
ii = mod(bsxfun(@plus, tmp1(:), reshape(kdy, [1, J, M])), K(1)) + 1; % [Q2, J, n] % row indices of the elements within each area,
                                                                   % whose leftmost element row indices are given above
jj = mod(bsxfun(@plus, reshape(tmp1.', [Q2, 1]), reshape(kdx, [1, J, M])), K(2)) + 1; % [Q2, J, M] % column indices ...
ll = reshape(bsxfun(@plus, reshape((jj - 1) * K(1), [Q2, 1, J, M]), reshape(ii, [Q2, J, 1, M])), [Q2, J2, M]);

% Duplicate values to have all the convolutions centered in the different elements
V = bsxfun(@times, reshape(v, [Q2, 1, M]), reshape(st.uu, [1, J2, M])); % [Q2, J2, M]
V(isnan(V(:))) = []; % [J2*M, 1] there are zeros in W at the positions where the
                 % measurements are missing, right size once the zeros are
                 % filtered

% Generate row indices (within G) [remark: Jt^2 nz elements per row]
kk = bsxfun(@times, ones(J2 * Q2, 1), 1:M); % [J2*Q2, M]
% try
G = sparse(kk(:), ll(:), V(:), M, K(1) * K(2));  % [M, prod(K)]
scale = st.sn;

end
