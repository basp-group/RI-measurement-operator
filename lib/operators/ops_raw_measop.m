function [measop, adjoint_measop, varargout] = ops_raw_measop(u,v, w, imsize, resolution_param, nufft_param)
% Generate the measurement op and its adjoint from a sampling pattern and
% user input settings
% operator (adapted from original code associated with
% :cite:p:`Fessler2003`).
%
% Parameters
% ----------
% u : double[1, d]
%     `u` coordinate in the spatial Fourier domain.
% v : double[1, d]
%    `v` coordinate in the spatial Fourier domain
% w : double[1, d]
%    `w` coordinate in the spatial Fourier domain
% imsize : double[1,1]
%     dimensions of the target image.
% resolution_param : struct
%     Structure containing user input on pixel resolution: either the pixel
%     size in arcsec ``resolution_param.pixelSize``  or the superresolution
%     factor ``resolution_param.superresolution``, ideally in the range [1.5, 2.5]. Default:
%     ``resolution_param.superresolution=1``.
% nufft_param : struct (optional)
%     Structure containing parameters of NUFFT
% Returns
% -------
% measop : function handle
%     Function handle for the measurement operator (de-gridding).
% adjoint_measop : function handle
%     Function handle for the adjoint operator (gridding).
% ri_normalization : double[1,1]
%    normalization factor in RI: peak of the PSF.
% varargout : cell
%    `` varargout{1}``: G matrix. `` varargout{2}``: grid-correction image

% Author: A. Dabbech 

%% [hardcoded] NUFFT parameters
if nargin < 6
    nufft_param.N = imsize; % image size
    nufft_param.J = [7, 7]; % kernel size
    nufft_param.K = 2 * imsize; % Fourier space size
    nufft_param.nshift = imsize / 2; % Fourier shift (matlab convention)
    nufft_param.ktype = 'minmax:kb'; % kernel type
end
%%  pixel resolution
% compute maximum projected baseline (nominal resolution)
maxProjBaseline = sqrt(max(u.^2 + v.^2));

% get user input
if isfield(resolution_param, 'pixelSize') && ~isempty(resolution_param.pixelSize)
     pixelSize = resolution_param.pixelSize;
     superresolution = (180 / pi) * 3600 / (pixelSize *  2 * maxProjBaseline);
           
     fprintf('\ninfo: user-specified pixelSize: %g arcsec (i.e. superresolution factor: %.3f) ', ....
          resolution_param.pixelSize,superresolution)

elseif  isfield(resolution_param, 'superresolution') && ~isempty(resolution_param.superresolution)
     superresolution = resolution_param.superresolution  ; 
     pixelSize = (180 / pi) * 3600 / ( superresolution*  2 * maxProjBaseline);
     fprintf('\ninfo: user-specified superresolution factor: %g (i.e. pixelSize: %g arcsec) ', ....
         superresolution, pixelSize)
else
     superresolution  = 1;
     pixelSize = (180 / pi) * 3600 / ( superresolution*  2 * maxProjBaseline);
     fprintf('\ninfo: default superresolution factor: %g (i.e. pixelSize: %g arcsec) ', ....
         superresolution, pixelSize)
end


%% define imaging spatial Fourier bandwidth
imagingBandwidth = maxProjBaseline * superresolution;


%% compute G matrix, associated scale parameter (gridding correction function), & Fourier operators
[Ft, IFt, G, scale] = op_nufft([-v, u].*(pi/imagingBandwidth), nufft_param.N, nufft_param.J, nufft_param.K, nufft_param.nshift);


%% check w-correction via w-projection
if ~isempty(w) && nnz(w)
    wproj_param.pixelSize = pixelSize; 
    wproj_param.CEnergyL2 = 1-1e-4; % [hardcoded] w-projection sparsity param
    wproj_param.GEnergyL2 = 1; % [hardcoded] w-projection sparsity param
    G = wprojection_nufft_mat(G, w, nufft_param, wproj_param);
end

%% define the measurememt operator & its adjoint
measop = @(x) ( G * Ft(x) ) ; 
adjoint_measop = @(y) IFt( G' * y );

%% additional output
if nargout == 3
    varargout{1} = {G};
elseif nargout == 4
    varargout{1} = {G};
    varargout{2} = {scale};
end


