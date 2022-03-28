function st = compute_interp_coeffs(om, Nd, Jd, Kd, n_shift, varargin)
% Compute the interpolation coefficients involved in the direct NUFFT
% operator (adapted from original code associated with
% :cite:p:`Fessler2003`).
%
% Parameters
% ----------
% om : double[:, d]
%     Non-uniform coordinates of the target of the data points in the
%     :math:`d`-dimensional Fourier domain.
% Nd : int[1, d]
%     Input tensor size along each dimension.
% Jd : int[1, d]
%     Size of the interpolation kernel along each dimension.
% Kd : int[1, d]
%     Size of the Fourier domain along each dimension.
% n_shift : int[1, d]
%     Phase-shift along each dimension in Fourier space (expressed in
%     number of samples in the Fourier domain).
% varargin : {string} or {string, cell, cell}
%     User-defined type of kernel, to be selected among the options
%     ``'minmax:kb'`` (used by default), ``'minmax:tuned'`` and
%     ``'kaiser'``. Possible options are documented below.
% ktype : ``'minmax:kb'``
%     Minmax interpolator with excellent KB scaling (``'minmax:kb'`` is
%     recommended, and used by default).
% ktype : ``'minmax:tuned'``
%     Minmax interpolator, somewhat numerically tuned.
% ktype : ``'kaiser'``
%     Kaiser-bessel (KB) interpolator (using default minmax best `alpha`,
%     `m`).
% ktype, alpha, m : ``'kaiser'``, double[:], double[:]
%     Kaiser-bessel (KB) interpolator using the specified values for the
%     parameters `alpha` and `m`).
%
% Returns
% -------
% st : struct
%     Structure containing the NUFFT scaling coefficients (``st.sn``) and
%     Auxiliary parameters to build the NUFFT interpolation matrix
%     (``st.alpha``, ``st.beta``, ``st.uu``).
%

% default/recommended interpolator is minmax with KB scaling factors
M = size(om, 1);
dd = length(Nd);

st.alpha = cell(dd, 1);
st.beta = cell(dd, 1);
is_kaiser_scale = false;
if ~isempty(varargin)
    ktype = varargin{1};
else
    ktype = 'minmax:kb';
end
st.ktype = ktype;

% set up whatever is needed for each interpolator
switch ktype
        % KB interpolator
    case 'kaiser'
        is_kaiser_scale = true;

        % with minmax-optimized parameters
        if length(varargin) == 1
            for id = 1:dd
                [st.kernel{id}, st.kb_alf(id), st.kb_m(id)] = ...
                    kaiser_bessel('inline', Jd(id));
            end

            % with user-defined parameters
        elseif length(varargin) == 3
            alpha_list = varargin{2};
            m_list = varargin{3};
            if (length(alpha_list) ~= dd) || (length(m_list) ~= dd)
                fail('#alpha=%d #m=%d vs dd=%d', ...
                    length(alpha_list), length(m_list), dd);
            end
            for id = 1:dd
                [st.kernel{id}, st.kb_alf(id), st.kb_m(id)] = ...
                    kaiser_bessel('inline', Jd(id), ...
                    alpha_list(id), m_list(id));
            end
        else
            fail 'kaiser should have no arguments, or both alpha and m';
        end

        % minmax interpolator with KB scaling factors (recommended default)
    case 'minmax:kb'
        for id = 1:dd
            [st.alpha{id}, st.beta{id}] = ...
                nufft_alpha_kb_fit(Nd(id), Jd(id), Kd(id));
        end

        % minmax interpolator with numerically "tuned" scaling factors
    case 'minmax:tuned'
        for id = 1:dd
            [st.alpha{id}, st.beta{id}, ok] = ...
                nufft_best_alpha(Jd(id), 0, Kd(id) / Nd(id));
            if ~ok; fail 'unknown J,K/N'; end
        end

    otherwise
        error('unknown kernel type %s', ktype);
end

st.tol = 0;

st.Jd = Jd;
st.Nd = Nd;
st.Kd = Kd;

st.M = M;
% st.om = om;

% scaling factors: "outer product" of 1D vectors
st.sn = 1;
if is_kaiser_scale
    % 'kaiser'
    for id = 1:dd
        nc = (0:Nd(id) - 1).' - (Nd(id) - 1) / 2;
        tmp = 1 ./ kaiser_bessel_ft( ...
            nc / Kd(id), Jd(id), st.kb_alf(id), st.kb_m(id), 1);
        st.sn = st.sn(:) * tmp';
    end
else
    % 'minmax:kb' and 'minmax:tuned'
    for id = 1:dd
        tmp = nufft_scale(Nd(id), Kd(id), st.alpha{id}, st.beta{id});
        st.sn = st.sn(:) * tmp';
    end
end
if length(Nd) > 1
    st.sn = reshape(st.sn, Nd); % [(Nd)]
else
    st.sn = st.sn(:); % [(Nd)]
end

% [J? M] interpolation coefficient vectors.  will need kron of these later
ud = cell(dd, 1);
for id = 1:dd
    N = Nd(id);
    J = Jd(id);
    K = Kd(id);
    if isfield(st, 'kernel')
        [c, arg] = ...
            nufft_coef(om(:, id), J, K, st.kernel{id}); % [J? M]
    else
        alpha = st.alpha{id};
        beta = st.beta{id};
        T = nufft_T(N, J, K, st.tol, alpha, beta); % [J? J?]
        [r, arg] = ...
            nufft_r(om(:, id), N, J, K, alpha, beta); % [J? M]
        c = T * r;  clear T r;
    end

    gam = 2 * pi / K;
    phase_scale = 1i * gam * (N - 1) / 2;

    phase = exp(phase_scale * arg); % [J? M] linear phase
    ud{id} = phase .* c; % [J? M]
end; clear c arg gam phase phase_scale N J K;

% build sparse matrix that is [M *Kd]
% with *Jd nonzero entries per frequency point
if dd >= 3
    tmp = prsod(Jd) * M * 8 / 10^9 * 2;
    if tmp > 1. % only display if more than 1GB
        printm('Needs at least %g Gbyte RAM', tmp);
    end
end

% kk = kd{1}; % [J1 M] % [PA] values of the indices locations
uu = ud{1}; % [J1 M] % [PA] values of the u coefficients
Jprod = Jd(1);
for id = 2:dd
    uu = reshape(uu, [Jprod, 1, M]) .* reshape(ud{id}, [1, Jd(id), M]); % ok
    Jprod = prod(Jd(1:id));
    uu = reshape(uu, Jprod, M);
end % now kk and uu are [*Jd M]

% apply phase shift
% pre-do Hermitian transpose of interpolation coefficients
phase = exp(1i * (om * n_shift(:))).'; % [1 M] % [PA] use of n_shift to shift the frequencies given in om
st.uu = conj(uu) .* phase(ones(1, prod(Jd)), :); % [*Jd M] % [PA] use bsxfun instead of duplicating entries...
