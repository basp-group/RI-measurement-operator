% Example script to simulate monochromatic RI data from a Fourier sampling pattern
clc; clear ; close all;
fprintf("*** Simulate radio data from a built-in astronomical image ***\n")

%% Setup path
addpath data;
addpath nufft;
addpath lib/operators;
addpath lib/ddes_utils/

%% ground truth image & settings
% image characteristics
imSize = [512, 512];
% simulations settings
simtype = 'realistic'; %  possible values: `realistic` ; `toy`
superresolution = 1; % ratio between imaged Fourier bandwidth and sampling bandwidth
%% observation setting : realistic / toy
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
        obsTime = 5;
        % obs. frequency in MHz
        frequency  = 1e9;
end
%% Fourier sampling pattern 
switch simtype
    case 'realistic'
        uvwdata = load(myuvwdatafile,'u','v','w');
        umeter =  uvwdata.u;
        vmeter =  uvwdata.v;
        wmeter =  uvwdata.w;
        clear uvwdata;
    case 'toy'
        % generate sampling pattern (uv-coverage)
        fprintf("\nsimulate Fourier sampling pattern using %s .. ", telescope)
        [umeter, vmeter, wmeter] = generate_uv_coverage(nTimeSamples, obsTime, telescope);
end
% convert in units of the wavelength
speedOfLight = 299792458;
u = umeter ./ (speedOfLight/frequency) ;
v = vmeter ./ (speedOfLight/frequency) ;
w = wmeter ./ (speedOfLight/frequency) ;
%% generate meas. op & its adjoint
fprintf("\nbuild NUFFT measurement operator .. ")
resolution_param.superresolution = superresolution; 
% resolution_param.pixelSize = [];
[measop, adjoint_measop] = ops_raw_measop(u, v, w, imSize, resolution_param);

% %% compute RI normalization factor  (just for info)
% dirac = sparse((imSize(1)/2)+1 , (imSize(2)/2)+1 , 1, imSize(1),imSize(2)) ;
% psf = real(adjoint_measop(measop(full(dirac))));
% ri_normalization = max(psf,[],'all');
