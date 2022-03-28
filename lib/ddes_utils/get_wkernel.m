function wkernel = get_wkernel(W, nTerm, PHASE, levelC)
%%-------------------------------------------------------------------------%
% Code: Arwa Dabbech.
% Last revised: [21/01/2022]
% -------------------------------------------------------------------------%
%%
%% w-term in the image domain, aka chirp
chirp  = exp(-2 * 1i * pi * W .* nTerm.lm); % chirp analytical expr. in the image space
nz = numel(chirp); % fft normalisation factor

%% w-kernel
chirpFourier = ifftshift(PHASE .* fft2(chirp .* nTerm.win)) ./ nz;

%%  sparsification of the chirp
[thresVal, ~] =  bisect(levelC, 1e-5, abs(chirpFourier), 2);
wkernel = full(chirpFourier .* (abs(chirpFourier) > thresVal));
end
