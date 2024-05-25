function nWimag = util_gen_imaging_weights(u, v, nW, N, param)
%
% Parameters
% ----------
% u : double[:]
%     u coordinate of the data point in the Fourier domain in radians.
% v : double[:]
%     v coordinate of the data points in the Fourier domain in radians.
% nW : double[:]
%     inverse of the noise s.t.d.
% N:  int[2]
%      image dimension
% param : struct
%     List of parameters to specify weights generation (can be omitted by
%     default). Fields: `weight_type` with values in {'uniform','briggs', 'none'}, `weight_gridsize`, `weight_robustness` that is the Briggs param.
%
% Returns
% -------
% nWimag : double[:]
%      weights inferred from the density of the sampling (uniform/Briggs).
%
% author: A. Dabbech, updated [25/05/2024]

%%
if ~isfield(param, 'weight_type')
    param.weight_type = 'uniform';
end
if ~isfield(param, 'weight_gridsize')
    param.weight_gridsize = 1;
end
if ~isfield(param, 'weight_robustness')
    param.weight_robustness = 0.0;
end

% number of meas.
nmeas = numel(u);

% size of the grid
N = floor(param.weight_gridsize*N);

% consider only half of the plane
u(v < 0) = -u(v < 0);
v(v < 0) = -v(v < 0);

% grid uv points
q = floor((u + pi)*N(2)/2/pi);
p = floor((v + pi)*N(1)/2/pi);

uvInd = sub2ind(N, p, q);
clear p q;

% Initialize gridded weights matrix with zeros
gridded_weights = zeros(N);

% inverse of the noise variance
nW2 = double(nW.^2);
if isscalar(nW2)
    nW2 = (nW2) .* ones(nmeas, 1);
end

% get gridded weights
for imeas = 1:nmeas
    switch param.weight_type
        case 'uniform'
            gridded_weights(uvInd(imeas)) = gridded_weights(uvInd(imeas)) + 1;
        case 'briggs'
            gridded_weights(uvInd(imeas)) = gridded_weights(uvInd(imeas)) + nW2(imeas);
    end

end

% Apply weighting based on weighting_type
switch param.weight_type
    case 'uniform'
        nWimag = 1 ./ sqrt(gridded_weights(uvInd));
    case 'briggs'
        % Compute robust scale factor
        robust_scale = (sum(gridded_weights, "all") / sum(gridded_weights.^2, "all")) * (5 * 10^(-param.weight_robustness)).^2;
        nWimag = 1 ./ sqrt(1+robust_scale.*gridded_weights(uvInd));
end

end
