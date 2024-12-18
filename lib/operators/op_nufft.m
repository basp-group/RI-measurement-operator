function [A, At, Gw, scale] = op_nufft(p, N, Nn, No, Ns, ktype)
% Create the nonuniform gridding matrix and fft operators (based on
% ``createG``).
%
% Parameters
% ----------
% p : double[:, 2]
%     Non-uniformly distributed frequency location points.
% N : int[2]
%     Size of the reconstruction image.
% Nn : int[2]
%     Size of the kernels (number of neighbors considered on each
%     direction).
% No : int[2]
%     Oversampled fft from which to recover the non uniform fft via kernel
%     convolution.
% Ns : int[2]
%     FFFT shift.
% ktype : string
%     Selected interpolator (among ``'minmax:kb'``, ``'minmax:tuned'`` and
%     ``'kaiser'``, see :mat:func:`nufft.compute_interp_coeffs_extended`
%     for associated kernel documentation). Recommended default is
%     ``'minmax:kb'``.
%
% Returns
% -------
% A : Function handle
%     Function handle for direct operator.
% At : function handle
%     Function handle for adjoint operator.
% Gw : double[:, :]
%     Global convolution kernel matrix.
% scale : double[:, :]
%     Scale paremters for the oversampled FFT.
%

%% compute the overall gridding matrix and its associated kernels
if ~exist('ktype', 'var')
    ktype = 'minmax:kb';
end

% st = nufft_init(p, N, Nn, No, Ns, ktype);
% scale = st.sn;

[Gw, scale] = createG(p, Nn, N, No, Ns, ktype);

% if isa(st.p, 'fatrix2')
%     error('fatrix2 has some very weird bugs with subindexing; force st.p to be a (sparse) matrix');
% end

%% operator function handles
[A, At] = op_nu_so_fft2(N, No, scale);

% whole G is stored in st.p
% Gw = st.p;

end
