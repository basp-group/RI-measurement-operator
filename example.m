% Comparison between Fessler's nufft code (default kb-minimax) with a
% trimmed down version of it (available in the nufft folder).

clc; clear all; close all;

%% Setup path
addpath data;
addpath nufft;
addpath irt;
addpath tests;
addpath lib/operators;
setup;

atol = 1e-8;
rtol = 1e-5;

%% NUFFT parameters
N = [2560, 1536];
J = [7, 7];
K = 2 * N;
nshift = N / 2;
ktype = 'minmax:kb';
T = 100;
cov_type = 'vlaa';
dl = 2.;
hrs = 5;

%% Error comparisons with Fessler's NUFFT implementation
% generate test frequencies
[u_ab, v_ab, na] = generate_uv_coverage(T, hrs, dl, cov_type);
M = na * (na - 1) / 2;
om = [v_ab(:), u_ab(:)];

%%
% compute G matrix and associated scale parameter
st = nufft_init(om, N, J, K, nshift, ktype);

%%
[G, scale, st2] = createG(om, J, N, K, nshift, ktype);

% discrepancy for G
err_G = norm(G - st.p, 'fro');
disp(isclose(nonzeros(G), nonzeros(st.p), atol, rtol));

% discrepancy for the scale parameter
err_scale = norm(scale(:) - st.sn(:), 2);
disp(isclose(scale(:), st.sn(:), atol, rtol));

%% Example of application of the NUFFT using the proposed interface
[A, At, G, scale] = op_nufft(om, N, J, K, nshift);

% direct NUFFT
x = rand(N);
nufft_x = G*A(x);

% adjoint NUFFT
y = (rand(size(om, 1), 1) + 1i*rand(size(om, 1), 1));
adj_nufft_y = At(G'*y);
