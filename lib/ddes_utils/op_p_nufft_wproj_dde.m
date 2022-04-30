function [A, At, G, W] = op_p_nufft_wproj_dde(nufft, p, w, nW, wproj, ddes, kernel)
% Create the nonuniform gridding matrix and fft operators to be used for
% parallel processing
%
% in:
% p{:}[2] - nonuniformly distributed frequency location points for each
%           cell member which will be treated in parallel
% N[2]    - size of the reconstruction image
% Nn[2]   - size of the kernels (number of neighbors considered on each direction)
% No[2]   - oversampled fft from which to recover the non uniform fft via
%           kernel convolution
% Ns[2]   - fft shift
%
% out:
% A[@]          - function handle for direct operator
% At[@]         - function handle for adjoint operator
% G{:}[:][:]    - convolution kernel matrix (small) associated with each
%               patch in the fourier plane
% W{:}          - mask of the values that contribute to the convolution
% ktype : string
%     Selected interpolator (among ``'minmax:kb'``, ``'minmax:tuned'`` and
%     ``'kaiser'``, see :mat:func:`nufft.compute_interp_coeffs_extended`
%     for associated kernel documentation). Recommended default is
%     ``'minmax:kb'``.

%%
% A. Onose, A. Dabbech, Y. Wiaux - An accelerated splitting algorithm for radio-interferometric %imaging: when natural and uniform weighting meet, MNRAS 2017, arXiv:1701.01748
% https://github.com/basp-group/SARA-PPD

%%
% flags
if ~exist('ddes', 'var'); dde_flag = 0;
elseif isempty(ddes); dde_flag = 0;
else; dde_flag = 1;
end

if ~exist('wproj', 'var');  wproj.measop_flag_wproj = 0;
end

if ~isfield(wproj, 'measop_flag_wproj'); wproj_flag = 0;
else; wproj_flag = wproj.measop_flag_wproj;
end

if ~isfield(nufft, 'ktype')
    nufft.ktype = 'minmax:kb';
end

% Fourier operators
[A, At, ~, ~] = op_nufft([0, 0], nufft.N, nufft.Nn, nufft.No, nufft.Ns, nufft.ktype);

% init.
G = [];
W = [];
if nargin > 1
    R = size(p, 1);

    % get de-gridding matrix
    if ~isempty(p)

        %% compute small gridding matrices associated with each parallel block
        G = cell(R, 1);
        W = cell(R, 1);

        % block start position
        fprintf('\nComputing block matrices ...\n');
        for q = 1:R

            tstart = tic;
            b_l = length(p{q});

            %% compute the small gridding matrix and its associated kernels
            [~, ~, Gw, ~] = op_nufft([p{q, 1}, p{q, 2}], nufft.N, nufft.Nn, nufft.No, nufft.Ns, nufft.ktype);

            %% now trim the zero rows and store a mask in W
            % preallocate W for speed
            W{q} = false(nufft.No(1) * nufft.No(2), 1);
            Gw = spdiags(nW{q}, 0, b_l, b_l) * Gw;

            %% check if w correction is needed
            effBandwidthWterm = max(abs(max(wproj.FoVy, wproj.FoVx) .* abs(w{q})));
            if effBandwidthWterm > 4 * max(wproj.uGridSize, wproj.vGridSize) % hard coded limit of the w bandwidth
                fprintf('\nWARNING:W-correction is needed ..\n');
            end

            %% update G: perform w-correction and incorporate ddes if available
            if wproj_flag || dde_flag
                wproj.paddFourierFactor = nufft.No ./ nufft.N;
                wproj.gImDims = nufft.N;
                wproj.supportK = prod(nufft.Nn);
                Gw = Gw.';
                if dde_flag && wproj_flag
                    Gw = update_G_ddes(Gw, w{q}, ddes{q}, wproj);
                elseif ~dde_flag && wproj_flag
                    Gw = update_G_ddes(Gw, w{q}, [], wproj);
                elseif dde_flag
                    Gw = update_G_ddes(Gw, [], ddes{q}, wproj);
                end
                Gw = Gw.';
            end

            %% check if eack line is entirely zero
            W{q} = any(abs(Gw), 1).';

            %% store only what we need from G
            G{q} = Gw(:, W{q});

            %% timing
            tend = toc(tstart);
            fprintf('Block matrix %d: %ds \n', q, ceil(tend));
        end
    end
end
end
