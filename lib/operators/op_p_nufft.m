function [A, At, G, W] = op_p_nufft(p, N, Nn, No, Ns, ww, param, ktype)
% Create the nonuniform gridding matrix and fft operators to be used for
% parallel processing (relying on ``op_nufft_irt``).
%
% Parameters
% ----------
% p : cell{2, 1} of double[:]
%     Non-uniformly distributed frequency location points for each cell
%     member which will be treated in parallel.
% N : int[2]
%     Size of the reconstruction image.
% Nn : int[2]
%     Size of the kernels (number of neighbors considered in each
%     direction).
% No : int[2]
%     Oversampled fft from which to recover the non uniform fft via kernel
%     convolution.
% Ns : int[2]
%     FFT shift.
% ww : cell{:} of bool[:]
%     Masks identifying data blocks.
% param : struct
%     Parameters to generate the NUFFT interpolation matrix.
% ktype : string
%     Selected interpolator (among ``'minmax:kb'``, ``'minmax:tuned'`` and
%     ``'kaiser'``, see :mat:func:`nufft.compute_interp_coeffs_extended`
%     for associated kernel documentation). Recommended default is
%     ``'minmax:kb'``.
%
% Returns
% -------
% A : function handle
%     Function handle for direct operator.
% At : function handle
%     Function handle for adjoint operator.
% G : cell{:} of complex[:, :]
%     Convolution kernel matrix (small) associated with patch in the
%     Fourier plane.
% W : cell{:} of int[:]
%     Mask of the values that contribute to the convolution.
%

%%
if ~exist('param', 'var')
    param = struct();
end
if ~isfield(param, 'use_nufft_blocks'); param.use_nufft_blocks = 1; end
if ~isfield(param, 'gen_only_fft_op'); param.gen_only_fft_op = 0; end
if ~exist('ww', 'var')
    ww = cell(length(p), 1);
    for q = 1:length(p)
        ww{q} = ones(length(p{q}(:, 1)), 1);
    end
end
if ~exist('ktype', 'var')
    ktype = 'minmax:kb';
end

R = size(p, 1);
if param.gen_only_fft_op
    [A, At, ~, ~] = op_nufft([0, 0], N, Nn, No, Ns, ktype);
    G = [];
    W = [];
    Gw = [];
else
    if ~param.use_nufft_blocks
        %% compute the overall gridding matrix and its associated kernels
        [A, At, Gw, ~] = op_nufft(cell2mat(p), N, Nn, No, Ns, ktype);

        %% compute small gridding matrices associated with each parallel block
        G = cell(R, 1);
        W = cell(R, 1);

        % block start position
        fprintf('\nComputing block matrices ...\n');
        b_st = 1;
        for q = 1:R
            tstart = tic;
            % current block length
            % the matrix Gw is structured identical to the structure of p
            % thus we grab it block by block
            b_l = length(p{q});

            % get a block out of the large G and trim it
            Gw(b_st:b_st + b_l - 1, :) = spdiags(ww{q}, 0, b_l, b_l) * Gw(b_st:b_st + b_l - 1, :);
            Gb = Gw(b_st:b_st + b_l - 1, :);

            %% now trim the zero rows and store a mask in W
            % check if eack line is entirely zero
            W{q} = any(Gb, 1).';

            % store only what we need from G
            G{q} = Gb(:, W{q});

            % iterate among the blocks
            b_st = b_st + b_l;
            tend = toc(tstart);
            fprintf('Block matrix %d: %ds \n', q, ceil(tend));
        end
    else

        %% compute small gridding matrices associated with each parallel block
        G = cell(R, 1);
        W = cell(R, 1);

        % block start position
        fprintf('\nComputing block matrices ...\n');
        for q = 1:R

            tstart = tic;
            b_l = length(p{q});

            %% compute the small gridding matrix and its associated kernels
            [~, ~, Gb, ~] = op_nufft([p{q, 1} p{q, 2}], N, Nn, No, Ns, ktype);

            %% now trim the zero rows and store a mask in W

            % preallocate W for speed
            W{q} = false(No(1) * No(2), 1);

            Gb = spdiags(ww{q}, 0, b_l, b_l) * Gb;

            % check if eack line is entirely zero
            W{q} = any(Gb, 1).';

            % store only what we need from G
            G{q} = Gb(:, W{q});

            tend = toc(tstart);
            fprintf('Block matrix %d: %ds \n', q, ceil(tend));
        end

        [A, At, ~, ~] = op_nufft([0, 0], N, Nn, No, Ns, ktype);
    end
end

end
