function G = wprojection_nufft_mat(G, w, nufft_param, wproj_param)

%% input
% G: sparse matrix; the de-gridding matrix
% w: vector; w-coordinates
% wproj_param : struct
%     Structure containing user input on the sparsity parameters of w-proj
% nufft_param : struct 
%     Structure containing parameters of NUFFT
%% output
% G:  de-gridding matrix incorporating the w correction 
%%-------------------------------------------------------------------------%
% Code: Arwa Dabbech.
% Last revised: [25/03/2024]
% -------------------------------------------------------------------------%
%% Global vars, Image & data dims
imSize = nufft_param.N; % image size
imFourierSize = nufft_param.K; % zero-padded image size
nufftKernelNumel = prod(nufft_param.J); % nufft kernel support
paddFourierFactor = unique(nufft_param.N ./ nufft_param.K ); % oversampling factor

% FoV details
FoV =  sin(wproj_param.pixelSize * imSize * pi / 180 / 3600) ; % Field of view
uvPixelSize = 1 ./ (paddFourierFactor .* FoV);
uvHalfBandwidth = max(uvPixelSize .* imSize); % half imaged BW

% check if w-correction is required
effectiveBandwidthWterm = max(FoV) * max(abs(w));
if effectiveBandwidthWterm > 4 * max(uvPixelSize) % hard coded limit of the w bandwidth
    fprintf('\ninfo:w-correction is enabled ..\n');
else
%     return;
end

%% w-projection 
% sparsity params
if isfield(wproj_param,'CEnergyL2')
    levelC = wproj_param.CEnergyL2; % energy thresholds for the w-kernel (should be  in [0.9,1[)
else
    levelC = 1 - 1e-4;
end

if isfield(wproj_param,'GEnergyL2')
    levelG = wproj_param.GEnergyL2; % energy thresholds for the w-kernel (should be  in [0.9,1[)
else
    levelG = 1 - 1e-4;
end

% wkernel bins
[WResolutionBin, PHASE, nTerm] = get_details_wkernel(w, FoV, imSize, paddFourierFactor, uvHalfBandwidth);

%% prepare data for sparse conv.
% restructure NUFFT kernels
nmeas = numel(w); % number of meas.
[rGNUFFT1d, ~, vGNUFFT] = find(transpose(G));
rGNUFFT1d = reshape(rGNUFFT1d, nufftKernelNumel, nmeas).';
vGNUFFT = reshape(vGNUFFT, nufftKernelNumel, nmeas).';
[rGNUFFT, cGNUFFT, ~] = ind2sub(imFourierSize, rGNUFFT1d);
clear rGNUFFT1d;
rGNUFFT = rGNUFFT.';
cGNUFFT = cGNUFFT.';
vGNUFFT = vGNUFFT.';

%% send data to labs
PHASE_CST = parallel.pool.Constant(PHASE);
nTerm_CST = parallel.pool.Constant(nTerm);
clear PHASE nTerm;

%% init.
tStart = tic;
dCol = cell(nmeas, 1);
dVal = cell(nmeas, 1);
dRow = cell(nmeas, 1);

%% compute de-gridding matrix
parfor wrow = 1:nmeas
    % build chirp &  convolve with ddes if available
    if WResolutionBin(wrow) > 1
        wkernel = get_wkernel(w(wrow), nTerm_CST.Value{WResolutionBin(wrow)}, PHASE_CST.Value{WResolutionBin(wrow)}, levelC);
        % sparse convolution
        nufft_kernel = [];
        shiftedPos = shift_ind([rGNUFFT(:, wrow), cGNUFFT(:, wrow)], imFourierSize(1), imFourierSize(2)); %#ok<PFBNS>
        nufft_kernel.dim = imFourierSize;
        nufft_kernel.i = shiftedPos(:, 1);
        nufft_kernel.j = shiftedPos(:, 2);
        nufft_kernel.a = vGNUFFT(:, wrow);
        shiftedPos = [];
        full_kernel = sconv2_modified(nufft_kernel, wkernel, 'same');
        % sparsify the G kernel
        if levelG < 1.0 && numel(wkernel) > 1
            [thresVal, ~] = bisect(levelG, 1e-5, abs(nonzeros(full_kernel)), 2);
            full_kernel = full_kernel .* (abs(full_kernel) > thresVal);
        end        
        [posShift1, posShift2, dVal{wrow}] = find(full_kernel);
        % ifftshift
        posOrig = shift_ind([posShift1, posShift2], imFourierSize(1), imFourierSize(2));
    else
        dVal{wrow} = vGNUFFT(:, wrow);
        posOrig = [rGNUFFT(:, wrow), cGNUFFT(:, wrow)];
    end
    
    GKernelNnz =  numel(dVal{wrow}) ; 
    dCol{wrow} = sub2ind(imFourierSize, posOrig(:, 1), posOrig(:, 2));
    dRow{wrow} = wrow * ones(GKernelNnz, 1);

end; clear PHASE_CST nTerm_CST rGNUFFT vGNUFFT cGNUFFT;

timeElapsed = toc(tStart);

%% build updated G mat
dCol = cell2mat(dCol);
dRow = cell2mat(dRow);
dVal = cell2mat(dVal);
G = sparse(dRow, dCol, dVal, nmeas, prod(imFourierSize));
clear dCol dRow dVal;
end
