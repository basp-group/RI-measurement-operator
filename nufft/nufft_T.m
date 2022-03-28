function T = nufft_T(N, J, K, tol, alpha, beta, use_true_diric)
% Precompute the matrix ``T = [C' S S' C]\inv`` used in NUFFT. This can be 
% precomputed, being independent of frequency location.
%
% Parameters
% ----------
% N : int[:]
% 	Signal length.
% J : int[:]
% 	Number of neighbors.
% K : int[:]
% 	FFT length.
% tol : double
% 	Tolerance for smallest eigenvalue.
% alpha : double[L+1, 1]
% 	Fourier coefficient vector for scaling.
% beta : double[:]
% 	Scale ``gamma = 2 * pi / K`` by this for Fourier series.
% use_true_diric : bool
% 	Use true Diric function (default is to use sinc approximation).
%
% Returns
% -------
% T : double[J, J]
%   Precomputed matrix.
%

% Author: Jeff Fessler, University of Michigan
% 
% Note
% ----
% Original code taken from :cite:p:`Fessler2003`, available at https://github.com/JeffFessler/mirt.
%

%%
%| Copyright 2001-12-8, Jeff Fessler, University of Michigan

% if nargin == 1 && strcmp(N, 'test'), nufft_T_test, return, end
if ~exist('tol', 'var') || isempty(tol)
	tol = 1e-7;
end
if ~exist('beta', 'var') || isempty(beta)
	beta = 1/2;
end
if ~exist('use_true_diric', 'var') || isempty(use_true_diric)
	use_true_diric = false;
end

% if N > K, fail 'N > K', end
if N > K
    error('N > K');
end

% default with unity scaling factors
if ~exist('alpha', 'var') || isempty(alpha)

	% compute C'SS'C = C'C
	[j1 j2] = ndgrid(1:J, 1:J);
	cssc = nufft_diric(j2 - j1, N, K, use_true_diric);

% Fourier-series based scaling factors
else
	if ~isreal(alpha(1)), error('need real alpha_0'), end
	L = length(alpha) - 1; % L
	cssc = zeros(J,J);
	[j1 j2] = ndgrid(1:J, 1:J);
	for l1 = -L:L
		for l2 = -L:L
			alf1 = alpha(abs(l1)+1);
			if l1 < 0, alf1 = conj(alf1); end
			alf2 = alpha(abs(l2)+1);
			if l2 < 0, alf2 = conj(alf2); end

			tmp = j2 - j1 + beta * (l1 - l2);
			tmp = nufft_diric(tmp, N, K, use_true_diric);
			cssc = cssc + alf1 * conj(alf2) * tmp;
%		printm('%d %d %s %s', l1, l2, num2str(alf1), num2str(alf2))
		end
	end
end


% Inverse, or, pseudo-inverse

%smin = svds(cssc,1,0);
smin = min(svd(cssc));
if smin < tol % smallest singular value
% 	warn('Poor conditioning %g => pinverse', smin)
	T = pinv(cssc, tol/10);
else
	T = inv(cssc);
end


% nufft_T_test
function nufft_T_test
N = 128; K = 2*N;
alpha = [1 0 0];
beta = 1/2;
for J=1:8
	T0 = nufft_T(N, J, K, [], alpha, beta, 0);
	T1 = nufft_T(N, J, K, [], alpha, beta, 1);
	sprintf('J=%d K/N=%d cond=%g %g', J, K/N, cond(T0), cond(T1))
end
