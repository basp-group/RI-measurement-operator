function GNUFFTW = update_G_ddes(GNUFFT, W, ddes, param)

%% input
% GNUFFT: transpose of the de-gridding matrix
% W: w-coordinates
% ddes: available antenna gains for each row

%% output
% GNUFFTW: transpose of the de-gridding matrix incorporating the w correction and ddes (if available)
%%-------------------------------------------------------------------------%
% Code: Arwa Dabbech.
% Last revised: [21/01/2022]
% -------------------------------------------------------------------------%
%%
%% flags of additional convolution kernels
flag_dde = 0;
flag_wproj = 0;
% flag wproj
if ~isempty(W); flag_wproj = 1;
end
% flag ddes
if ~isempty(ddes); flag_dde = 1;
end

%% Global vars, Image & data dims
paddFourierFactor = param.paddFourierFactor;
gImDims = param.gImDims; % image dims
ImFourierDims = paddFourierFactor .* gImDims;
supportNufft = param.supportK;
nMeas = size(GNUFFT, 2);

if ~isempty(W) && flag_wproj
    FoV = [param.FoVy param.FoVx]; % Field of view
    uvPixelSize = 1 ./ (paddFourierFactor .* FoV);
    uvHalfBW = max(uvPixelSize .* gImDims); % half imaged BW

    levelC = param.CEnergyL2; % energy thresholds for the w-kernel (should be  in [0.9,1[)
    levelG = param.GEnergyL2; % energy thresholds for  the full kernel (should be in [0.9,1[)

    % wkernel bins
    [WResolutionBin, PHASE, nTerm] = get_details_wkernel(W, FoV, gImDims, paddFourierFactor, uvHalfBW);

else % defined for compilation purposes in parfor
    levelG = 1;
    levelC = 1;
    PHASE = cell(nMeas, 1);
    nTerm = cell(nMeas, 1);
    WResolutionBin = sparse(nMeas, 1);
    if isempty(W); W = sparse(nMeas, 1);
    end
end

if ~flag_dde
    % defined for compilation purposes in parfor
    ddes = sparse(nMeas, 1);
end
ddeSupport = [sqrt(size(ddes, 2)), sqrt(size(ddes, 2))];

%% prepare data for sparse conv.
% restructure NUFFT kernels
[rGNUFFT1d, ~, vGNUFFT] = find(GNUFFT);
rGNUFFT1d = reshape(rGNUFFT1d, supportNufft, nMeas).';
vGNUFFT = reshape(vGNUFFT, supportNufft, nMeas).';
[rGNUFFT, cGNUFFT, ~] = ind2sub(ImFourierDims, rGNUFFT1d);
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
dummyCol = cell(nMeas, 1);
dummyVal = cell(nMeas, 1);
dummyRow = cell(nMeas, 1);

%% compute de-gridding matrix
parfor wrow = 1:nMeas
    ddekernel = 1;
    if flag_wproj
        % build chirp &  convolve with ddes if available
        if WResolutionBin(wrow) > 1
            ddekernel = get_wkernel(W(wrow), nTerm_CST.Value{WResolutionBin(wrow)}, PHASE_CST.Value{WResolutionBin(wrow)}, levelC);
            if flag_dde; ddekernel = conv2(ddekernel, reshape(ddes(wrow, :), ddeSupport(1), ddeSupport(2)), 'full'); %#ok<PFBNS>
            end
        end
    elseif flag_dde
        % get dde kernel
        ddekernel = (reshape(ddes(wrow, :), ddeSupport(1), ddeSupport(2)));
    end

    if numel(ddekernel) > 1
        % sparse convolution
        nufft_kernel = [];
        shiftedpos = shift_ind([rGNUFFT(:, wrow), cGNUFFT(:, wrow)], ImFourierDims(1), ImFourierDims(2)); %#ok<PFBNS>
        nufft_kernel.dim = ImFourierDims;
        nufft_kernel.i = shiftedpos(:, 1);
        nufft_kernel.j = shiftedpos(:, 2);
        nufft_kernel.a = vGNUFFT(:, wrow);
        shiftedpos = [];
        full_kernel = (sconv2_modified(nufft_kernel, ddekernel, 'same'));
    else
        full_kernel = sparse(rGNUFFT(:, wrow), cGNUFFT(:, wrow), vGNUFFT(:, wrow), ImFourierDims(1), ImFourierDims(2));
    end

    % sparsify the G kernel
    if levelG < 1.0 && numel(ddekernel) > 1
        [thresVal, ~] = bisect(levelG, 1e-5, abs(nonzeros(full_kernel)), 2);
        full_kernel = full_kernel .* (abs(full_kernel) > thresVal);
    end
    SupportGKernel = nnz(full_kernel);

    % ifftshift
    [posShift1, posShift2, dummyVal{wrow}] = find(full_kernel);
    if numel(ddekernel) > 1
        posOrig = shift_ind([posShift1, posShift2], ImFourierDims(1), ImFourierDims(2));
    else
        posOrig = [posShift1, posShift2];
    end

    dummyCol{wrow} = sub2ind(ImFourierDims, posOrig(:, 1), posOrig(:, 2));
    dummyRow{wrow} = wrow * ones(SupportGKernel, 1);

end; clear PHASE_CST nTerm_CST rGNUFFT vGNUFFT cGNUFFT;

timeElapsed = toc(tStart);

%% build updated G mat
dummyCol = cell2mat(dummyCol);
dummyRow = cell2mat(dummyRow);
dummyVal = cell2mat(dummyVal);
GNUFFTW = sparse(dummyCol, dummyRow, dummyVal, prod(ImFourierDims), nMeas);

end
