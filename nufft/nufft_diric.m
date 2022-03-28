function f = nufft_diric(k, N, K, use_true_diric)
% "Regular fourier" Dirichlet-function WITHOUT phase
% ``nufft_diric(t) = sin(pi N t / K) / ( N * sin(pi t / K) )
% \approx sinc(t / (K/N))``
%
% Parameters
% ----------
% k : double[...]
% 	Sample locations (unitless real numbers)
% N : int[...]
% 	Signal length.
% K : int[...]
% 	FFT length.
% use_true_diric : bool
% 	Use true Diric function (default is to use sinc approximation).
%
% Returns
% -------
% f : double[...]
% 	Corresponding function values.
%
% Warning
% -------
% Caution: 
% - matlab's version is different: ``sin(N * x / 2) / (N * sin(x / 2))``
% - ``nufft_diric()`` is ``K``-periodic for odd ``N`` but ``2K``-periodic 
% for even ``N``.
%
% Note
% ----
% Original code taken from :cite:p:`Fessler2003`, available at https://github.com/JeffFessler/mirt.
%

% Author: Jeff Fessler, University of Michigan
%

%%
%| Copyright 2001-12-8, Jeff Fessler, University of Michigan
% if nargin == 1 && strcmp(k, 'test'), nufft_diric_test, return, end

if nargin < 4
	use_true_diric = false;
end

% diric version
if use_true_diric
	t = (pi/K) * k;
	f = sin(t);
	i = abs(f) > 1e-12; % nonzero denominator
	f(i) = sin(N*t(i)) ./ (N * f(i));
	f(~i) = sign(cos(t(~i)*(N-1)));

% sinc version
else
	f = nufft_sinc(k / (K/N));
end


% function nufft_diric_test
% Nlist = [2^3 2^5 2^3-1];
% Klist = 2*Nlist; Klist(end) = Nlist(end);
% jf pl 3 1
% for ii=1:length(Nlist)
% 	N = Nlist(ii);
% 	K = Klist(ii);
% 
% 	kmax = 2*K+4;
% 	kf = linspace(-1,1,2001) * kmax; % fine grid
% 	ki = (-kmax:kmax);
% 
% 	gf = nufft_diric(kf, N, K, 1);
% 	gi = nufft_diric(ki, N, K, 1);
% 	sf = nufft_diric(kf, N, K, 0);
% 	if exist('diric', 'file') == 2 % matlab's diric is in signal toolbox
% 		dm = diric((2*pi/K)*kf,N);
% 		jf_equal(gf, dm)
% 	%	printm('max %% difference vs matlab = %g', max_percent_diff(gf,dm))
% 	else
% 		dm = zeros(size(kf));
% 	end
% 	jf('sub', ii)
% %	plot(kf, gf, 'y-', kf, sf, 'c-', kf, dm, 'r--', ki, gi, 'y.')
% 	plot(kf, gf, 'r-', kf, sf, 'b-', ki, gi, 'r.')
% 	axis tight
% 	xtick([-2:2]*K)
% %	legend('nufft diric', 'sinc', 'matlab diric')
% 	legend('nufft diric', 'sinc')
% 	xlabelf('$k$'), ylabelf('diric($k$)')
% 	printm('max %% difference vs sinc = %g', max_percent_diff(gf,sf))
% 	titlef('N = %d, K = %d', N, K)
% end
