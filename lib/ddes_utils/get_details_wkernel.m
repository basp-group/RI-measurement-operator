function [WcoorBin, PHASE, nterm] = get_details_wkernel(W, FoV, imSize, paddFourierFactor, uvHalfBW)
%%-------------------------------------------------------------------------%
% Author: Arwa Dabbech.
% Last revised: [21/01/2022]
% -------------------------------------------------------------------------%
%%
woversampling = 2;
ImFourierDim = paddFourierFactor .* imSize;
eRatioWTermPrBsln = woversampling .* abs(sin(max(FoV)) .* W) ./ (uvHalfBW); % ratio between the wterm BW and the imaged BW
eRatioLowerBound = 4 * max(1 ./ max(ImFourierDim)); % tiny -->no correction
dimBINS = eRatioLowerBound:eRatioLowerBound:0.5;
WcoorBinPos = cell(length(dimBINS), 1);
WcoorBin = zeros(length(W), 1);
WcoorBinPos{1} = find(eRatioWTermPrBsln <= eRatioLowerBound);
WcoorBin(WcoorBinPos{1}) = 1;
max_bin = 1;
w_bin_bound = [];
for bin = 2:length(dimBINS)
    WcoorBinPos{bin} = find(eRatioWTermPrBsln .* (eRatioWTermPrBsln > dimBINS(bin - 1)) .* (eRatioWTermPrBsln <= dimBINS(bin)));
    if nnz(WcoorBinPos{bin})
        w_bin_bound(bin, 1) = dimBINS(bin) * (uvHalfBW) ./ abs(sin(max(FoV))) ./ woversampling;
        max_bin = bin;
    else; w_bin_bound(bin, 1) = 0;
    end
    WcoorBin(WcoorBinPos{bin}) = bin;
end
if nnz(~WcoorBin)
    fprintf('\nWarning: %d non binned w --> will be assigned the largest bin', nnz(~WcoorBin));
    WcoorBinPos{bin} = [WcoorBinPos{bin}; find(~WcoorBin)];
    WcoorBin(~WcoorBin) = length(dimBINS);
end
dimBINS = dimBINS(1:max_bin);
%% w kernel in the image domain
NCurr(:, 1)  =  (floor(imSize(1) .* dimBINS(:)) - mod(floor(imSize(1) .* dimBINS(:)), 2));
NCurr(:, 2)  =  (floor(imSize(2) .* dimBINS(:)) - mod(floor(imSize(2) .* dimBINS(:)), 2));
NCurr = max(NCurr, [4 4]);
nterm = cell(max_bin, 1);
PHASE =  cell(max_bin, 1);
for binW = 2:max_bin
    if w_bin_bound(binW, 1)
        sanitycheck = w_bin_bound(binW, 1) >= max(abs(W(WcoorBin == binW)));
        if ~sanitycheck; error('w-term: resolution bining went wrong !!');
        end
        w_bin_bound(binW, 1) = max(abs(W(WcoorBin == binW)));
        w_bin_bound_(binW, 1) = min(abs(W(WcoorBin == binW)));
    end
    if ~isempty(WcoorBinPos{binW})
        % build the lm grid
        Nt_SCurr =  paddFourierFactor .* NCurr(binW, :);
        [l_SSCurr, m_SSCurr] = meshgrid(-Nt_SCurr(2) / 2:Nt_SCurr(2) / 2 - 1, -Nt_SCurr(1) / 2:Nt_SCurr(1) / 2  - 1);
        dl_SCurr = 2 * sin(paddFourierFactor(1) .* FoV(1) * 0.5) / Nt_SCurr(1); % assuming same Fov in both directions ..
        dm_SCurr = 2 * sin(paddFourierFactor(2) .* FoV(2) * 0.5) / Nt_SCurr(2);
        l_SSCurr = l_SSCurr .* dl_SCurr;
        m_SSCurr = m_SSCurr .* dm_SCurr;
        % build the n-term (later used to build the chirp)
        nterm{binW}.lm  = sqrt(1 - l_SSCurr.^2 - m_SSCurr.^2) - 1;
        nshiftSCurr =  NCurr(binW, :) ./ paddFourierFactor;
        [fxSCurr, fySCurr] = meshgrid((0:(Nt_SCurr(1) - 1)) / Nt_SCurr(1), (0:(Nt_SCurr(2) - 1)) / Nt_SCurr(2).');
        omSCurr = -2 * 1i * pi * ([fySCurr(:), fxSCurr(:)]);
        phaseSCurr = exp(omSCurr * nshiftSCurr(:)).';
        PHASE{binW}  = reshape(phaseSCurr, Nt_SCurr);
        % get window
        % cols
        half_fov = NCurr(binW, :) / 2;
        win_fun = @blackman;
        frac = 2;
        win = window(win_fun, round(half_fov(2) * frac));
        wc = win(:);
        if mod(numel(wc), 2)
            pos = find(wc == 1);
            if ~pos;     slice = wc(1:(numel(wc) - 1) / 2);
            else; slice = wc(1:pos - 1);
            end
        else;    slice = wc(1:numel(wc) / 2);
        end
        wc = [zeros(half_fov(2) - numel(slice), 1); slice; ones(Nt_SCurr(2) / 2, 1); flipud(slice); zeros(half_fov(2) - numel(slice), 1)];
        % rows
        win = window(win_fun, round(half_fov(1) * frac));
        wr = win(:);
        if mod(numel(wr), 2)
            pos = find(wr == 1);
            if ~pos;     slice = wr(1:(numel(wr) - 1) / 2);
            else; slice = wr(1:pos - 1);
            end
        else;    slice = (wr(1:(numel(wr)) / 2));
        end
        wr = [zeros(half_fov(1) - numel(slice), 1); slice; ones(Nt_SCurr(1) / 2, 1); flipud(slice); zeros(half_fov(1) - numel(slice), 1)];
        % 2D  window
        [maskr, maskc] = meshgrid(wc, wr);
        nterm{binW}.win = maskr .* maskc;
    else
        PHASE{binW} = [];
        nterm{binW} = [];
    end

end
