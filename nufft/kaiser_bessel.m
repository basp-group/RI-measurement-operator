function [kb, alpha, kb_m] = kaiser_bessel(x, J, alpha, kb_m, K_N)
% Generalized Kaiser-Bessel function for ``x`` in support [-J/2,J/2]
% shape parameter "alpha" (default 2.34 J)
% order parameter "kb_m" (default 0)
% see (A1) in lewitt:90:mdi, JOSA-A, Oct. 1990
%
% Parameters
% ----------
% x : double[M, 1]
% 	Arguments.
% J : int[:]
% 	Number of neighbors.
% alpha :
% 	Parameters of the interpolation kernel.
% kb_m :
% 	Parameters of the interpolation kernel.
% K_N :
% 	Parameters of the interpolation kernel.
%
% Returns
% -------
% kb : [M, 1]
% 	KB function values, if ``x`` is numbers, or ``string`` for 
%   ``kernel(k,J)``,  if ``x`` is a ``string``, or inline function, if 
%   ``x`` is ``inline``.
% alpha
% 	Parameters of the interpolation kernel.
% kb_m
% 	Parameters of the interpolation kernel.
%

% Example
% -------
% >>> [kb, alpha, kb_m] = kaiser_bessel(x, J, alpha, kb_m)
% >>> [kb, alpha, kb_m] = kaiser_bessel(x, J, 'best', 0, K_N)
%
% Note
% ----
% Original code taken from :cite:p:`Fessler2003`, available at https://github.com/JeffFessler/mirt.
%

% Author: Jeff Fessler, University of Michigan
%

%%
%| Copyright 2001-3-30, Jeff Fessler, University of Michigan

% Modification 2002-10-29 by Samuel Matej
% - for Negative & NonInteger kb_m the besseli() function has
%	singular behavior at the boundaries - KB values shooting-up/down
%	(worse for small alpha) leading to unacceptable interpolators
% - for real arguments and higher/reasonable values of alpha the
%	besseli() gives similar values for positive and negative kb_m
%	except close to boundaries - tested for kb_m=-2.35:0.05:2.35
%	(besseli() gives exactly same values for integer +- kb_m)
%	=> besseli(kb_m,...) approximated by besseli(abs(kb_m),...), which
%	behaves well at the boundaries
% WARNING: it is not clear how correct the FT formula (JOSA) is
%	for this approximation (for NonInteger Negative kb_m)
% NOTE: Even for the original KB formula, the JOSA FT formula
%	is derived only for m > -1 !

% if nargin < 1, help(mfilename),error(mfilename), end

% if nargin == 1 && strcmp(x, 'test') % example plots
% 	J = 8; alpha = 2.34 * J;
% 	x = linspace(-(J+1)/2, (J+1)/2, 1001)';
% %	x = linspace(J/2-1/4, J/2+1/4, 1001)';
% 
% 	mlist = [-4 0 2 7];
% 	leg = {};
% 	for ii=1:length(mlist)
% 		kb_m = mlist(ii);
% 		yy(:,ii) = kaiser_bessel(x, J, alpha, kb_m);
% 		func = kaiser_bessel('inline', 0, alpha, kb_m);
% 		yf = func(x, J);
% 		if any(yf ~= yy(:,ii)),
% 		[yf yy(:,ii)]
% 		error 'bug', end
% 		leg{ii} = sprintf('m=%d', kb_m);
% 	end
% 	yb = kaiser_bessel(x, J, 'best', [], 2);
% 	plot(	x, yy(:,1), 'c-', x, yy(:,2), 'y-', ...
% 		x, yy(:,3), 'm-', x, yy(:,4), 'g-', x, yb, 'r--')
% 	leg{end+1} = 'best';
% 	axis tight, legend(leg)
% %	axisy(0, 0.01)	% to see endpoints
% 	xlabelf '\kappa'
% 	ylabelf 'F(\kappa)'
% 	titlef('KB functions, J=%g \alpha=%g', J, alpha)
% return
% end

if ~isvarname('J'), J = 6; end
if ~exist('alpha', 'var') || isempty(alpha), alpha = 2.34 * J; end
if ~exist('kb_m', 'var') || isempty(kb_m), kb_m = 0; end

if ischar(alpha)
	[alpha kb_m] = kaiser_bessel_params(alpha, J, K_N); % ok
end

if ischar(x)
	if ischar(alpha)
		if ~isvarname('K_N'), error 'K_N required', end
		kb = 'kaiser_bessel(k, J, ''%s'', [], %g)';
		kb = sprintf(kb, alpha, K_N);
	else
		kernel_string = 'kaiser_bessel(k, J, %g, %g)';
		kb = sprintf(kernel_string, alpha, kb_m);
	end
	if strcmp(x, 'inline')
		kb = inline(kb, 'k', 'J');
	elseif ~strcmp(x, 'string')
		error '1st argument must be "inline" or "string"'
	end
return
end

%
% Warn about use of modified formula for negative kb_m
%
if (kb_m < 0) && ((abs(round(kb_m)-kb_m)) > eps)
	persistent warned
	if isempty(warned)	% print this reminder only the first time
		sprintf('\nWarning: Negative NonInt kb_m=%g in kaiser_bessel()', kb_m)
		sprintf('	- using modified definition of KB function\n')
		warned = 1;
	end
end

kb_m_bi = abs(kb_m);		% modified "kb_m" as described above
ii = abs(x) < J/2;
f = sqrt(1 - (x(ii)/(J/2)).^2);
denom = besseli(kb_m_bi,alpha);
if ~denom
	printf('m=%g alpha=%g', kb_m, alpha)
end
kb = zeros(size(x));
kb(ii) = f.^kb_m .* besseli(kb_m_bi, alpha*f) / denom;
kb = real(kb);


%
% optimized shape and order parameters
%
function [alpha, kb_m] = kaiser_bessel_params(alpha, J, K_N)
if strcmp(alpha, 'best')
	if K_N ~= 2
		warning 'kaiser_bessel optimized only for K/N=2'
		sprintf('using good defaults: m=0 and alpha = 2.34*J')
		kb_m = 0;
		alpha = 2.34 * J;
	else
		kb_m = 0; % hardwired, because it was nearly the best!
		try
            s = 'kaiser,m=0.mat';
			s = load('kaiser,m=0.mat');
			ii = find(J == s.Jlist);
			if isempty(ii)
				[~, ii] = min(abs(J - s.Jlist));
				warning('J=%d not found, using %d', J, s.Jlist(ii))
			end
			alpha = J * s.abest.zn(ii);
		catch
			warning(['could not open file "' s '" so using default alpha = 2.34 J which should be fine.'])
			alpha = 2.34 * J;
		end
	end
else
	error 'unknown alpha mode'
end
