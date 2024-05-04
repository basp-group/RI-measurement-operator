function nWimag = util_gen_imaging_weights(u, v, nW2, N, param)
%
% Parameters
% ----------
% u : double[:]
%     u coordinate of the data point in the Fourier domain in radians.
% v : double[:]
%     v coordinate of the data points in the Fourier domain in radians.
% nW2 : double[:]
%     inverse of the variance.
% N:  int[2]
%      image dimension
% param : struct
%     List of parameters to specify weights generation (can be omitted by
%     default). Fields: `weight_type` with values in {'uniform','robust', 'none'}, `weight_gridsize`, `weight_robustness` that is the Briggs param.
%
% Returns
% -------
% nWimag : double[:]
%      weights inferred from the density of the sampling (uniform/Briggs).
%
% author: A. Dabbech

%%
if ~isfield(param, 'weight_type'); param.weight_type = 'uniform'; end
if ~isfield(param, 'weight_gridsize'); param.weight_gridsize = 1; end
if ~isfield(param, 'weight_robustness'); param.weight_robustness = 0.0; end

nmeas = numel(u);
N = param.weight_gridsize * N;

% Initialize gridded weights matrix with zeros
p = mod(floor((v(:)+pi)* N(1)/(2*pi)), N(1))  + 1; 
q = mod(floor((u(:)+pi)* N(2)/(2*pi)), N(2))  + 1; 

gridded_weights = zeros(N);

for imeas = 1:nmeas
    switch param.weight_type
        case 'uniform'
            gridded_weights(p(imeas), q(imeas)) = gridded_weights(p(imeas), q(imeas)) + 1;
        case  'robust'
            gridded_weights(p(imeas), q(imeas)) = gridded_weights(p(imeas), q(imeas)) + nW2(imeas);
    end
end

% Compute robust scale factor
switch param.weight_type
    case 'robust'
        robust_scale = (sum(gridded_weights(:)) / sum(gridded_weights(:).^2)) * (5 * 10^(-param.weight_robustness)).^2;
end

nWimag = ones(nmeas, 1);
% Apply weighting based on weighting_type
for imeas = 1:nmeas
    switch param.weight_type
        case 'uniform'
            nWimag(imeas) = 1/ sqrt(gridded_weights(p(imeas), q(imeas)));
        case  'robust'
            nWimag(imeas) = 1 / sqrt(1 + robust_scale * gridded_weights(p(imeas), q(imeas)));
    end
end


end

