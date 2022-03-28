function x = so_fft2_adj(X, N, No, scale)
% Computes the inverse scaled FTT2 involved in the adjoint NUFFT operator.
%
% Parameters
% ----------
% X : complex[:]
%     2D fourier transform of the image.
% N : int[2]
%     Size of image.
% No : int[2]
%     Size of zero-padded image.
% scale : double[:, :]
%     Scale factor precomputed by ``nufft_init``, used to cancel the
%     scaling performed asscoiated with the convolution kernel to the non-
%     uniform frequency domain.
%
% Returns
% -------
% x : complex[:]
%     Inverse scaled 2D FFT coefficients involved in the adjoint NUFFT
%     operator.
%

%%
iscale = conj(scale);

% compute the inverse fourier transform
X = reshape(X, No);
x = ifft2(X);

% % scale the solution
% x = No(1) * No(2) * x(:);
%
% % reshape to oversampled image size
% x = reshape(x, No);

% trim oversampled part to actual image size
x = x(1:N(1), 1:N(2));

% scale the solution
% x = No(1) * No(2) * x;

% rescale
x = (No(1) * No(2)) * (x .* iscale);

end
