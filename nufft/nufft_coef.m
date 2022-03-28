def  nufft_coef(om, J, K, kernel):
    """_summary_

    
    """


return a

function [coef, arg] = nufft_coef(om, J, K, kernel)
% NUFFT interpolation coefficient vectors for a given kernel function.
%
% Parameters
% ----------
% om : double[M, 1]
%     Digital frequency omega in radians.
% J : int
%     Number of neighbors used per frequency location.
% K : _type_
%     FFT size (should be >= N, the signal length).
% kernel : function handle
%     Kernel function handle.
%
% Returns
% -------
% coef : double[J, M]
%     Coef. vector for each frequency.
% arg : double[J, M]
%     Kernel argument.
%
% Note
% ----
% Original code taken from :cite:p:`Fessler2003`, available at https://github.com/JeffFessler/mirt.
%

% Author: Jeff Fessler, University of Michigan
%

%%
%| Copyright 2002-4-11, Jeff Fessler, University of Michigan

if nargin < 4, ir_usage, end

M = length(om);
gam = 2*pi/K;
dk = om / gam - nufft_offset(om, J, K);	% [M 1]
arg = outer_sum(-[1:J]', dk');			% [J M] kernel arg

coef = feval(kernel, arg, J);			% [J M]
