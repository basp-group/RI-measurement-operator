function k0 = nufft_offset(om, J, K)
% Offset for NUFFT.
%
% Parameters
% ----------
% om : double[M, 1]
% 	omega (radians), typically in :math:`[-\pi, \pi)` (not essential!)
% J : int[1, d]
% 	Number of neighbors used for NUFFT interpolation.
% K : int[1, d]
% 	FFT size along each domension.
%
% Returns
% -------
% k0 : int[M,1]
% 	Prepared for ``mod(k0 + [1:J], K)`` (``+ 1`` for matlab).
%
% Note
% ----
% Original code taken from :cite:p:`Fessler2003`, available at https://github.com/JeffFessler/mirt.
%

% Author: Jeff Fessler, University of Michigan
%

%%
%| Copyright 2000-1-9, Jeff Fessler, University of Michigan

% if nargin < 3, ir_usage, end
gam = 2*pi/K;
k0 = floor(om / gam - J/2); % new way

return

% old way:
if mod(J,2) % odd J
	k0 = round(om / gam) - (J+1)/2;

else % even J
	k0 = floor(om / gam) - J/2;
end


% compare old and new version of nufft_offset
% they match perfectly for even J
% they match for odd J too for nonnegative arguments,
% but do not match at -0.5, -1.5, etc., because round()
% rounds *away* from zero which means rounding up for
% positive arguments but rounding down for negative arguments.

% function nufft_offset_test
% for J=3:4
% 	x = linspace(-4,4,81);
% 
% 	% old way
% 	if mod(J,2) % odd J
% 		y = round(x) - (J+1)/2;
% 	else % even J
% 		y = floor(x) - J/2;
% 	end
% 
% 	% new way
% 	z = floor(x - J/2);
% 	plot(x, y, 'c.', x, z, 'yo')
% 
% 	pr x(find(z ~= y))
% end
