function X = so_fft2(x, No, scale)
% Computes and oversampled and scaled FFT2 with the scale factors
% precomputed from nufft_init
%
% in:
% x[:][:]  - input image
% No[2]    - overscale size
% scale[:] - scale parameters precomputed by nufft_init
%
% out:
% X[:]     - FFT2 coefficients
%%
%A. Onose, A. Dabbech, Y. Wiaux - An accelerated splitting algorithm for radio-interferometric %imaging: when natural and uniform weighting meet, MNRAS 2017, arXiv:1701.01748
%https://github.com/basp-group/SARA-PPD
%%

% apply scaling factors
x = x .* scale; 

% oversampled FFT 
X = fft2(x, No(1), No(2));
X = X(:);

end