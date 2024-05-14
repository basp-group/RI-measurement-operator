% Example script to simulate RI data, using a toy Fourier sampling pattern

clc; clear ; close all;
fprintf("*** Simulate toy radio data from a ground-truth astronomical image ***\n")

%% Setup paths
addpath data;
addpath nufft;
addpath lib/operators;
addpath lib/utils;
addpath lib/ddes_utils;

%% simulation setting: realistic / toy
simtype = 'realistic'; % possible values: `realistic` ; `toy`
noiselevel = 'drheuristic'; % possible values: `drheuristic` ; `inputsnr`
superresolution = 1.5; % ratio between imaged Fourier bandwidth and sampling bandwidth

% optional functionalities
weighting_on = true;  % enable imaging weights: briggs (aka robust) / uniform
weight_load = false;  % load weights from input .mat file
wproj_on = false; % wprojection 

% obs. setting
switch simtype
    case 'realistic'
        myuvwdatafile = 'tests/test.mat';
        frequency = load(myuvwdatafile,'frequency').frequency;
    case 'toy'
        % antenna configuration
        telescope = 'vlaa';
        % total number of snapshots
        nTimeSamples = 100;
        % obs duration in hours
        obsTime = 4;
        % obs. frequency in MHz
        frequency  = 1e9;
        % default
        weight_load = false; 
end
%% ground truth image 
fprintf("\nread ground truth image  .. ")
gdthim = fitsread('tests/3c353_gdth.fits'); 
imSize = size(gdthim);
% display
figure(1), imagesc(gdthim), colorbar, ...
title('ground-truth image'), axis image,  axis off,

%% data noise settings
switch noiselevel
    case 'drheuristic'
        % dynamic range of the ground truth image
        targetDynamicRange = 1e4; 
    case 'inputsnr'
         % user-specified input signal to noise ratio
        isnr = 40; % in dB
end

%% Fourier sampling pattern
switch simtype
    case 'realistic'
        fprintf("\nload Fourier sampling pattern .. ")
        uvwdata = load(myuvwdatafile,'u','v','w');
        umeter =  uvwdata.u;
        vmeter =  uvwdata.v;
        wmeter =  uvwdata.w;
        clear uvwdata;
        % for info 
        try nominalPixelSize = double(load(myuvwdatafile,'nominal_pixelsize').nominal_pixelsize);
        end
     
    case 'toy'       
        % generate sampling pattern (uv-coverage)
        fprintf("\nsimulate Fourier sampling pattern using %s .. ", telescope)
        [umeter, vmeter, wmeter] = generate_uv_coverage(nTimeSamples, obsTime, telescope);
end

% convert uvw in units of the wavelength
speedOfLight = 299792458;
u = umeter ./ (speedOfLight/frequency) ;
v = vmeter ./ (speedOfLight/frequency) ;
w = wmeter ./ (speedOfLight/frequency) ;

% maximum projected baseline (just for info)
maxProjBaseline  = sqrt(max(u.^2+v.^2));

%% generate meas. op & its adjoint
fprintf("\nbuild measurement operator .. ")
resolution_param.superresolution = superresolution;
% resolution_param.pixelSize = nominalPixelSize/superresolution;

nufft_param.N = imSize; % image size
nufft_param.J = [7, 7]; % kernel size
nufft_param.K = 2 * imSize; % Fourier space size
nufft_param.nshift = imSize / 2; % Fourier shift (matlab convention)
nufft_param.ktype = 'minmax:kb'; % kernel type

weighting_param.gen_weights =  weighting_on && (~weight_load);
weighting_param.weight_type = 'briggs';
weighting_param.weight_robustness = 0.0;

wproj_param.enable_wproj = false;
wproj_param.CEnergyL2 = 1 - 1e-4; %  w-projection sparsity param
wproj_param.GEnergyL2 = 1; % w-projection sparsity param
 
[measop, adjoint_measop, nWimag] = ops_raw_measop(u, v, w, imSize, ....
    resolution_param, weighting_param, nufft_param, wproj_param);

if weighting_on && weight_load
    switch simtype
        case 'realistic'
            load(myuvwdatafile,'nWimag');
            nWimag = double(nWimag(:));
    end
end

%% model clean visibilities 
fprintf("\nsimulate model visibilities .. ")
vis = measop(gdthim);

%number of data points
nmeas = numel(vis);

%% model data

% noise vector
switch noiselevel
    case 'drheuristic'
        fprintf("\ngenerate noise (noise level commensurate of the target dynamic range) .. ")
        if weighting_on 
            % include weights in the measurement op.
            measop_1 = @(x) (nWimag.*measop(x));
            adjoint_measop_1 = @(x) (adjoint_measop(nWimag.*x));
            measopSpectralNorm_1 = op_norm(measop_1, @(y) real(adjoint_measop_1(y)), imSize, 10^-4, 500, 0);

            measop_2 = @(x) ((nWimag.^2) .* measop(x));
            adjoint_measop_2 = @(x) (adjoint_measop((nWimag.^2).*x));
            measopSpectralNorm_2 = op_norm(measop_2, @(y) real(adjoint_measop_2(y)), imSize, 10^-4, 500, 0);

            % correction factor
            eta_correction = sqrt(measopSpectralNorm_2/measopSpectralNorm_1);

            % noise standard deviation heuristic
            tau  = sqrt(2 * measopSpectralNorm_1) / targetDynamicRange /eta_correction;
        else
            % compute measop spectral norm to infer the noise heuristic
            measopSpectralNorm = op_norm(measop, @(y) real(adjoint_measop(y)), imSize, 10^-4, 500, 0);
            eta_correction = 1;
            % noise standard deviation heuristic
            tau  = sqrt(2 * measopSpectralNorm) / targetDynamicRange ;
        end
        
        % noise realization(mean-0; std-tau)
        noise = tau * (randn(nmeas,1) + 1i * randn(nmeas,1))./sqrt(2);

        % input signal to noise ratio
        isnr = 20 *log10 (norm(vis)./norm(noise));
        fprintf("\ninfo: random Gaussian noise with input SNR: %.3f db", isnr)

    case 'inputsnr'
        fprintf("\ngenerate noise from input SNR  .. ")
        % user-specified input signal to noise ratio
        tau = norm(vis) / (10^(isnr/20)) /sqrt( (nmeas + 2*sqrt(nmeas)));
        noise = tau * (randn(nmeas,1) + 1i * randn(nmeas,1))./sqrt(2);
end

% data
fprintf("\nget data vector .. ")
y = vis + noise;

%% back-projected data
fprintf("\nget (non-normalised) back-projected data  .. ")
if weighting_on
    dirty = real( adjoint_measop((nWimag.^2).*y) );
    figure, imagesc(dirty), colorbar, title ('dirty image (weights applied)'), axis image,   axis off,
else
    dirty = real( adjoint_measop(y) );
    figure, imagesc(dirty), colorbar, title ('dirty image'), axis image,   axis off,
end

%% generate input data file for uSARA/AIRI/R2D2 imager  (just for info)
% whitening vector
fprintf("\nsave data file  .. ")
mkdir 'results'
matfilename = "results/3c353_data.mat" ;
dirtyfilename = "results/3c353_dirty.fits" ; 

% save mat file
nW = tau *ones(nmeas,1);
save(matfilename, "y", "nW", "u", "v","w","maxProjBaseline","frequency",'-v7.3')

% add imaging weights
if weighting_on
    save(matfilename,"nWimag",'-append')
end

% save (non-normalised) dirty image
fitswrite(dirty, dirtyfilename)


fprintf('\nDone.')
