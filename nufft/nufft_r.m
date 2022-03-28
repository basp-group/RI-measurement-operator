function [rr, arg] = nufft_r(om, N, J, K, alpha, beta, use_true_diric)
% Make NUFFT ``r`` vector.
% 
% Parameters
% ----------
% om : double[M, d]
% 	Digital frequency omega in radians.
% N : int[1, d]
% 	Signal length.
% J : int[1, d]
% 	Number of neighbors used per frequency location.
% K : int[1, d]
% 	FFT size (should be ``> N``)
% alpha : complex[0:L]
% 	Fourier series coefficients of scaling factors.
% beta : double[:]
%   Scale ``gamma=2pi./K`` by this in Fourier series, typically is ``K./N`` 
%   :cite:p:`Fessler2003` or 0.5 (Liu)
% use_true_diric : bool
% 	Optional flag for debugging purposes (defaults to false).
%
% Returns
% -------
% rr : double[J, M]
%   ``r`` vector for each frequency
% arg : double[J, M]
%   Dirac argument for ``t=0``.
%
%
% Note
% ----
% Original code taken from :cite:p:`Fessler2003`, available at https://github.com/JeffFessler/mirt.
%

% Author: Jeff Fessler, University of Michigan
%

%%
%| Copyright 2001-12-13, Jeff Fessler, University of Michigan

if ~exist('alpha', 'var') || isempty(alpha)
	alpha = 1; % default Fourier series coefficients of scaling factors
end
if ~exist('beta', 'var') || isempty(beta)
	beta = 0.5; % default is Liu version for now
end
if ~exist('use_true_diric', 'var') || isempty(use_true_diric)
	use_true_diric = false;
end

M = length(om);
gam = 2*pi/K;
dk = om / gam - nufft_offset(om, J, K); % [M 1]
% arg = outer_sum(-(1:J)', dk');			% [J M] diric arg for t=0
arg = bsxfun(@plus, -(1:J)', dk');			% [J M] diric arg for t=0

L = length(alpha) - 1;
if ~isreal(alpha(1)), error('need real alpha_0'), end

if L > 0
	rr = zeros(J,M);
	for l1 = -L:L
		alf = alpha(abs(l1)+1);
		if l1<0, alf = conj(alf); end
		r1 = nufft_diric(arg + l1 * beta, N, K, use_true_diric);
		rr = rr + alf * r1;			% [J M]
	end
else
	rr = nufft_diric(arg, N, K, use_true_diric); % [J M]
end
