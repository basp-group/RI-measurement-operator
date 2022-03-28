function [A, At] = op_nu_so_fft2(N, No, scale)
% Oversampled ftt2 and scaled operator computed by a modified NUFFT
% function.
%
% Parameters
% ----------
% N : int[2]
%     Size of the reconstruction image.
% No : int[2]
%     Oversampled fft from which to recover the non uniform fft via kernel
%     convolution.
% Ns : int[2]
%     Fft shift.
%
% Returns
% -------
% A : function handle
%     Function handle for direct operator.
% At : function handle
%     Function handle for adjoint operator.
%

%%
A = @(x) so_fft2(x, No, scale);
At = @(x) so_fft2_adj(x, N, No, scale);

end
