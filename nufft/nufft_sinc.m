function y = nufft_sinc(x)
% J. Fessler's version of "sinc" function, because matlab's ``sinc()`` is 
% in a toolbox.
%
% Parameters
% ----------
% x : double[:, ...]
%     Input array.
%
% Returns
% -------
% y : double[:, ...]
%     Evaluation of ``sinc(x)``.
%
% Note
% ----
% Original code taken from :cite:p:`Fessler2003`, available at https://github.com/JeffFessler/mirt.
%

% Author: Jeff Fessler, University of Michigan
%

%%
%| Copyright 2001-12-8, Jeff Fessler, University of Michigan

if strcmp(x, 'test'), nufft_sinc_test, return, end

iz = find(x == 0); % indices of zero arguments
x(iz) = 1;
y = sin(pi*x) ./ (pi*x);
y(iz) = 1;


% test
% function nufft_sinc_test
% 
% x = linspace(-4, 4, 2^21+1)';
% 
% nufft_sinc(0); % warm up
% cpu etic
% y1 = nufft_sinc(x);
% cpu etoc 'nufft_sinc time'
% 
% if 2 == exist('sinc')
% 	sinc(0); % warm up
% 	cpu etic
% 	y2 = sinc(x);
% 	cpu etoc 'matlab sinc time'
% 	jf_equal(y1, y2)
% end
% 
% if im, plot(x, y1, '-'), end
