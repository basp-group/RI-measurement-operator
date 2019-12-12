% Comparison between Fessler's nufft code (default kb-minimax) with a 
% trimmed down version of it.
clc; clear all; close all;

%% Setup path
addpath data
addpath nufft
addpath irt
setup

%% NUFFT parameters
N = [2560, 1536];
J = [7,7];
K = 2*N;
nshift = N/2;
T = 100;
cov_type = 'vlaa';
dl = 2.;
hrs = 5;

%% Error comparisons
% generate test frequencies
[u_ab, v_ab, na] = generate_uv_coverage(T, hrs, dl, cov_type);
M = na*(na-1)/2;
om = [v_ab(:), u_ab(:)];

% compute G matrix and associated scale parameter
st = nufft_init(om, N, J, K, nshift);
[G, scale] = createG(om, J, N, K, nshift);

% discrepancy for G
err_G = norm(G - st.p, 'fro');

% discrepancy for the scale parameter
err_scale = norm(scale(:)-st.sn(:), 2);
