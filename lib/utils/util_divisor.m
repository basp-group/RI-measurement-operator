function d = util_divisor(n)
% Provides a list of integer divisors of a number.
%
% Parameters
% ----------
% n : int
%     Input integer value.
%
% Returns
% -------
% d : int[1, :]
%     Row vector of all distinct divisors of the positive integer `n`,
%     including 1 and `n`.
%
% Example
% --------
% >>> a = divisor(12);
% >>> a = [1, 2, 3, 4, 6, 12];
%
% Note
% ----
% - This function uses the default `factor()` routine in Matlab and hence
%   is limited to input values up to :math:`2^{32}`. However if `factor()`
%   routine does get updated for larger integers, this function will still
%   work fine.
%   Using `factor()` provides a significant speed improvement over manually
%   seaching for the each divisor of `n`.
% - Author: Yash Kochar ( yashkochar@yahoo.com )
% - Last modified: 21st June 2009.
%
% .. codeauthor:: Yash Kochar ( yashkochar@yahoo.com )
%
% See Also
% --------
% factor, primes
%

%% Input error check :
%   Check whether input is positive integer and scalar.
if ~isscalar(n)
    error('divisor:NonScalarInput', 'Input must be a scalar.');
end
if (n < 1) || (floor(n) ~= n)
  error('divisor:PositiveIntegerOnly', 'Input must be a positive integer.');
end

%% Find prime factors of number :
pf = factor(n);         % Prime factors of n
upf = unique(pf);       % Unique

%% Calculate the divisors
d = upf(1).^(0:1:sum(pf == upf(1)))';
for f = upf(2:end)
    d = d * (f.^(0:1:sum(pf == f)));
    d = d(:);
end
d = sort(d)';   % To further improve the speed one may remove this sort command
                % Just remember to take the transpose of "d" to get a result
                % as a row vector instead of a column vector.
