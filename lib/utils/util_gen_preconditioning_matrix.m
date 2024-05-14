function [aW] = util_gen_preconditioning_matrix(u, v, param)
% Generate the diagnoal preconditioning matrix (accounting for the sampling
% density in the Fourier domain, similarly to uniform weighting
% :cite:p:`Onose2017`).
%
% Parameters
% ----------
% u : double[:]
%     u coordinate of the data point in the Fourier domain.
% v : double[:]
%     v coordinate of the data points in the Fourier domain.
% param : struct
%     List of parameters to specify weights generation (can be omitted by
%     default).
%
% Returns
% -------
% aW : double[:]
%     Diagonal preconditioner (uniform weighting), encoded as a vector.
%

%%
if ~isfield(param, 'gen_uniform_weight_matrix'); param.gen_uniform_weight_matrix = 0; end
if ~isfield(param, 'uniform_weight_sub_pixels'); param.uniform_weight_sub_pixels = 1; end

um = u;
vm = v;
um(v<0) = -um(v<0);
vm(v<0) = -vm(v<0);

aWw = ones(length(vm), 1);

if param.gen_uniform_weight_matrix == 1
    Noy = param.uniform_weight_sub_pixels * param.Noy;
    Nox = param.uniform_weight_sub_pixels * param.Nox;

    lsv = linspace(-pi, pi, Noy + 1);
    lsu = linspace(-pi, pi, Nox + 1);
    [v_, sv] = sort(vm);

    for k = 1:Noy
        [sfv_l, sfv_h] = util_sort_find(v_, lsv(k), lsv(k + 1), k < Noy, k == Noy);
        sfv = sv(sfv_l:sfv_h);
        if ~isempty(sfv)
            [u_, su] = sort(um(sfv));
            for j = 1:Nox
                [sfu_l, sfu_h] = util_sort_find(u_, lsu(j), lsu(j + 1), j < Nox, j == Nox);
                sfu = su(sfu_l:sfu_h);
                if ~isempty(sfu)
                    aWw(sfv(sfu)) = length(sfu);
                end
            end
        end
    end
end

aW = 1 ./ aWw;

end
