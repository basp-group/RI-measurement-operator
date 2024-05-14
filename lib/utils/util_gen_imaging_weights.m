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
% author: A. Dabbech

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

nmeas = numel(u);
% size of the grid
N = floor(param.weight_gridsize*N);

% consider only half of the plane
u(v < 0) = -u(v < 0);
v(v < 0) = -v(v < 0);

% grid uv points
q = floor((u + pi)*N(2)/2/pi);
p = floor((v + pi)*N(2)/2/pi);

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
            gridded_weights(p(imeas), q(imeas)) = gridded_weights(p(imeas), q(imeas)) + 1;
        case 'briggs'
            gridded_weights(p(imeas), q(imeas)) = gridded_weights(p(imeas), q(imeas)) + nW2(imeas);
    end
end

% Compute robust scale factor
switch param.weight_type
    case 'briggs'
        robust_scale = (sum(gridded_weights(:)) / sum(gridded_weights(:).^2)) * (5 * 10^(-param.weight_robustness)).^2;
end

% init
nWimag = ones(nmeas, 1);

% Apply weighting based on weighting_type
for imeas = 1:nmeas
    switch param.weight_type
        case 'uniform'
            nWimag(imeas) = 1 / sqrt(gridded_weights(p(imeas), q(imeas)));
        case 'briggs'
            nWimag(imeas) = 1 / sqrt(1+robust_scale*gridded_weights(p(imeas), q(imeas)));
    end
end


end
