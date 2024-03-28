% Example script to simulate RI data, using a toy Fourier sampling pattern

% clc; clear ; close all;
fprintf("*** Simulate toy radio data from a built-in astronomical image ***\n")

%% Setup paths
addpath data;
addpath nufft;
addpath lib/operators;
addpath lib/utils;


%% simulation setting: realistic / toy
simtype = 'realistic'; % possible values: `realistic` ; `toy`
noiselevel = 'drheuristic'; % possible values: `drheuristic` ; `inputsnr`
superresolution = 1.5; % ratio between imaged Fourier bandwidth and sampling bandwidth

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
end
%% ground truth image 
fprintf("\nread ground truth image  .. ")
% built-in matlab image
gdthim = imread('ngc6543a.jpg') ; 
% crop region of interest
gdthim = double(gdthim(49:560, 27:538)); 
% normalize image (peak = 1)
gdthim = gdthim./max(gdthim,[],'all'); 
% characteristics
imSize = size(gdthim);
% display
figure(1), imagesc(gdthim), colorbar, title ('ground truth image'), axis image,  axis off,

%% data noise settings
switch noiselevel
    case 'drheuristic'
        % dynamic range of the ground truth image
        targetDynamicRange = 255; 
    case 'inputsnr'
         % user-specified input signal to noise ratio
        inputSNR = 40; % in dB
end

%% Fourier sampling pattern
switch simtype
    case 'realistic'
        fprintf("\nload Fourier sampling pattern .. ")
        uvwdata = load(myuvwdatafile,'u','v','w');
        umeter =  uvwdata.u;
        vmeter =  uvwdata.v;
        wmeter =  uvwdata.w;
        clear uvwdata
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
maxProjBaseline   = sqrt(max(u.^2+v.^2));

%% generate meas. op & its adjoint
fprintf("\nbuild NUFFT measurement operator .. ")
resolution_param.superresolution = superresolution; 
% resolution_param.pixelSize = nominalPixelSize/superresolution; 

[measop, adjoint_measop] = ops_raw_measop(u,v, w, imSize, resolution_param);

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
        % compute measop spectral norm to infer the noise heuristic
        measopSpectralNorm = op_norm(measop, @(y) real(adjoint_measop(y)), imSize, 1-6, 500, 0);
        % noise standard deviation heuristic
        tau  = sqrt(2 * measopSpectralNorm) / targetDynamicRange;
        % noise realization(mean-0; std-tau)
        noise = tau * (randn(nmeas,1) + 1i * randn(nmeas,1))./sqrt(2);
        % input signal to noise ratio
        inputSNR = 20 *log10 (norm(vis)./norm(noise));
        fprintf("\ninfo: random Gaussian noise with input SNR: %.3f db", inputSNR)

    case 'inputsnr'
        fprintf("\ngenerate noise from input SNR  .. ")
        % user-specified input signal to noise ratio
        tau = norm(vis) / (10^(inputSNR/20)) /sqrt( (nmeas + 2*sqrt(nmeas)));
        noise = tau * (randn(nmeas,1) + 1i * randn(nmeas,1))./sqrt(2);
end

% data
fprintf("\nsimulate data  .. ")
y = vis + noise;


%% back-projected data
fprintf("\nget back-projected data  .. ")
dirty = real( adjoint_measop(y) );

% display
figure(2), imagesc(dirty), colorbar, title ('(non-normalized) dirty image'), axis image,   axis off,
fprintf('\nDone.')


% %% compute RI normalization factor  (just for info)
% dirac = sparse((imSize(1)/2)+1 , (imSize(2)/2)+1 , 1, imSize(1),imSize(2)) ;
% psf = real(adjoint_measop(measop(full(dirac))));
% ri_normalization = max(psf,[],'all');
% 
%% generate input data file for uSARA/AIRI/R2D2 imager  (just for info)
% % whitening vector
% nW = tau *ones(nmeas,1);
% 
% mkdir 'results'
% % save mat file
% save("./results/ngc6543a_toy_data.mat", "y", "nW", "u", "v","w","maxProjBaseline","frequency",'-v7.3')