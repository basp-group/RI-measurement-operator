function [alphas, beta] = nufft_alpha_kb_fit(N, J, K, varargin)
% Return the alpha and beta corresponding to LS fit of L components
% to optimized Kaiser-Bessel scaling factors (``m=0``, ``alpha=2.34*J``).
% This is the best method known by J. Fessler currently for choosing alpha.
%
% Parameters
% ----------
% N : int[:]
% 	Signal length.
% J : int[:]
% 	Number of neighbors.
% K : int[:]
% 	FFT length.
% varargin : option, [1]
% 	Specify the ``Nmid`` midpoint: ``floor(N/2)`` or default 
%   ``(N-1)/2``
%
% Returns
% -------
% alphas : complex[:]
% 	Parameters of the interpolation kernel.
% beta : double[:]
% 	Parameters of the interpolation kernel.
%
% Note
% ----
% Original code taken from :cite:p:`Fessler2003`, available at https://github.com/JeffFessler/mirt.
%

% Author: Jeff Fessler, University of Michigan
%

%%
%| Copyright 2002-7-16, Jeff Fessler, University of Michigan

% if nargin < 3, ir_usage, end

arg.beta = 1;
arg.chat = 0;
arg.Nmid = (N-1)/2;
if N > 40
	arg.L = 13;		% empirically found to be reasonable
else
	arg.L = ceil(N/3);	% a kludge to avoid "rank deficient" complaints
end

% arg = vararg_pair(arg, varargin);

%kb_alf = 2.34 * J;	% KB shape parameter
%kb_m = 0;		% KB order

nlist = (0:(N-1))' - arg.Nmid;

% kaiser-bessel with previously numerically-optimized shape
[~, kb_a, kb_m] = kaiser_bessel('string', J, 'best', 0, K/N); % ok
kernel_ft = kaiser_bessel_ft('handle', J, kb_a, kb_m, 1); % should be ok
sn_kaiser = 1 ./ kernel_ft(nlist/K); % [N]


% use regression to match NUFFT with BEST kaiser scaling's
gam = 2*pi/K;
X = cos(arg.beta * gam * nlist * (0:arg.L)); % [N L]
% coef = regress(sn_kaiser, X)';
coef = (X \ sn_kaiser)'; % this line sometimes generates precision warnings
if any(isnan(coef(:))) % if any NaN then big problem!
	coef = (pinv(X) * sn_kaiser)'; % try pinv() istead of mldivide \
	if any(isnan(coef(:)))
		error('bug: NaN coefficients.  Ask JF for help!')
	end
end
alphas = [real(coef(1)) coef(2:end)/2];


if arg.chat
	sprintf('cond # for LS fit to KB scale factors: %g', cond(X))
	srintf('fit norm = %g', norm(X * coef' - sn_kaiser))
end

beta = arg.beta;
