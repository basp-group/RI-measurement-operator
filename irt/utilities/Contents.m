% UTILITIES
%
% Files
%   arg_get                - function [arg, ii] = arg_get(list, what, default)
%   arg_pair               - |function arg = arg_pair(varargin)
%   beeper                 - make beeping noise to alert that a job is done.
%   bspline_1d_coef        - function coef = bspline_1d_coef(fn, varargin)
%   bspline_1d_interp      - function ft = bspline_1d_interp(fn, ti, varargin)
%   bspline_1d_synth       - function ft = bspline_1d_synth(coef, ti, varargin)
%   caller_name            - |function [name, line] = caller_name(level)
%   col                    - function x = col(x)
%   complexify             - function y = complexify(x)
%   cpu                    - function out = cpu(arg, varargin)
%   deg2rad                - function r = deg2rad(d)
%   detex                  - function y = detex(x);
%   dft_sym_check          - |function dft_sym_check(xk)
%   diffc                  - |function y = diffc(x)
%   difff                  - function y = difff(x)
%   dims_same              - |function y = dims_same(a, b, varargin)
%   div0                   - function r = div0(num, den)
%   double6                - function y = double6(x)
%   doubles                - function d = doubles(s)
%   downsample1            - |function y = downsample1(x, down, varargin)
%   downsample2            - |function y = downsample2(x, m, varargin)
%   downsample3            - |function y = downsample3(x, m)
%   dsingle                - function y = dsingle(x)
%   embed                  - |function ff = embed(x, mask, varargin)
%   equivs                 - |function out = equivs(var1, var2, command)
%   fail                   - function fail(varargin)
%   filtmat                - function mat = filtmat('1d', kernx, nx)
%   fld_read               - function [data, coord, dims] = fld_read(file, options)
%   fld_write              - function fld_write(file, data, [options])
%   flipdims               - |function x = flipdims(x, varargin)
%   fractional_delay       - function y = fractional_delay(x, delay)
%   fwhm                   - function [fw, hr, hl] = fwhm(psfs, ii, doplot)
%   fwhm1                  - |function [fw, hr, hl] = fwhm1(psfs, options)
%   fwhm2                  - |function [fw, angle, rad] = fwhm2(psf, [options])
%   fwhm_match             - |function [fwhm_best, costs, im_best] = ...
%   gaussian_kernel        - function kern = gaussian_kernel(fwhm, nk_half)
%   group2d                - |function groups = group2d(nx, ny, ngv, mask, chat)
%   has_aspire             - function yn = has_aspire
%   has_mex_jf             - function yn = has_mex_jf
%   highrate_centers       - function centers = highrate_centers(data, L, M)
%   hist_bin_int           - |function [nk center] = hist_bin_int(data, varargin)
%   hist_equal             - |function [nk, center] = hist_equal(data, ncent, varargin)
%   ifft_sym               - function y = ifft_sym(varargin)
%   imax                   - function [ii, i2] = imax(a, flag)
%   imin                   - |function [ii i2] = imin(a, flag2d)
%   interp1_jump           - function yi = interp1_jump(xj, yj, xi, {arguments for interp1})
%   interp1_lagrange       - |function y = interp1_lagrange(xi, yi, x)
%   interp1x               - function yi = interp1x(x, y, xi, flag1)
%   ir_apply_tridiag_inv   - function output = ir_apply_tridiag_inv(sub, diags, sup, rhs)
%   ir_best_scale          - function scale = ir_best_scale(x, y)
%   ir_bspline_basis       - function out = ir_bspline_basis(t, k, ti)
%   ir_bspline_knot        - function knot = ir_bspline_knot(ti, k)
%   ir_chebyshev_poly      - function tk = ir_chebyshev_poly(n)
%   ir_conv                - function y = ir_conv(x, psf, varargin)
%   ir_dct2                - todo: comments etc
%   ir_dct8                - | 2D DCT of each 8x8 block
%   ir_dctmtx              - |function out = ir_dctmtx(siz)
%   ir_display_struct      - function ir_display_struct(ob, varargin)
%   ir_dot_double          - function dot = ir_dot_double(a, b)
%   ir_dwt_filters         - |function [coef codes] = ir_odwt1(x, varargin)
%   ir_fftshift2           - function y = ir_fftshift2(x)
%   ir_fwrite              - function ir_fwrite(file, array, type, varargin)
%   ir_has_imfilter        - function out = ir_has_imfilter
%   ir_idct2               - function x = ir_idct2(y)
%   ir_im2col              - function [blocks, idx] = ir_im2col(I, blkSize, stride)
%   ir_imfill1             - function yy = ir_imfill1(xx)
%   ir_imfilter_many       - function y = ir_imfilter_many(x, psf, varargin)
%   ir_interpft            - |function yy = ir_interpft(xx, K, dim)
%   ir_is_live_script      - function out = ir_is_live_script
%   ir_is_octave           - |function out = ir_is_octave
%   ir_iter_fld_write      - |function out = ir_iter_fld_write(x, iter, varargin)
%   ir_nargout             - |function out = ir_nargout(fun)
%   ir_odwt1               - function [coef codes] = ir_odwt1(x, varargin)
%   ir_odwt2               - function [coef codes] = ir_odwt2(x, varargin)
%   ir_pad_into_center     - function xpad = ir_pad_into_center(x, npad, varargin)
%   ir_patch_avg           - function [out weight] = ir_patch_avg(patches, idx, dim, varargin)
%   ir_pet_blank_unmash    - ir_pet_blank_unmash.m
%   ir_poly2_fun           - function [fun, d1, d2] = ir_poly2_fun(order, [options])
%   ir_project_k_sparse    - function y = ir_project_k_sparse(x, k)
%   ir_project_simplex     - function x = ir_project_simplex(y)
%   ir_read_mat            - function x = ir_read_mat(file, pick)
%   ir_read_op             - function test = ir_read_op(dir, file, slice, frame, chat)
%   ir_reclass             - |function y = ir_reclass(x, newclass)
%   ir_sparse              - function out = ir_sparse(in)
%   ir_str2func            - function h = ir_str2func(string)
%   ir_struct_find_field   - |function [out ok] = ir_struct_find_field(ss, field)
%   ir_unwrap              - function ph = ir_unwrap(ph, varargin)
%   ir_usage               - function ir_usage(mfile_name)
%   ir_webread             - function st = ir_webread(urlsuff, varargin)
%   is_pre_v7              - |function y = is_pre_v7
%   isfreemat              - function out = isfreemat
%   isvar                  - function tf = isvar(name, varargin)
%   jf                     - function out = jf(varargin)
%   jf_assert              - function jf_assert(command)
%   jf_dcm_write           - |function jf_dcm_write(data, dcm_name, WindowCenter, WindowWidth, varargin)
%   jf_equal               - |function yn = jf_equal(a, b, varargin)
%   jf_histn               - |function [hist center] = jf_histn(data, varargin)
%   jf_pair_parse          - |function [out ii] = jf_pair_parse(cells, str, varargin)
%   jf_protected_names     - function pn = jf_protected_names
%   jf_whos_nan            - script to find variables in work space that have nan values
%   jinc                   - function y = jinc(x)
%   kde_pmf1               - |function [pmf xs] = kde_pmf1(x, varargin)
%   kde_pmf2               - function [pmf xs ys] = kde_pmf2(x, y, varargin)
%   kde_pmf_width          - function dx = kde_pmf_width(x, varargin)
%   lloyd_max_hist         - function centers = lloyd_max_hist(data, centers, MM, tol, max_iter, chat)
%   load_ascii_skip_header - |function data = load_ascii_skip_header(file)
%   mag_angle_real         - function [mag, ang] = mag_angle_real(x, cutoff)
%   masker                 - function y = masker(x, mask)
%   max_percent_diff       - |function d = max_percent_diff(s1, s2, [options])
%   min_cos_quad           - function t = min_cos_quad(m, p, b, c, niter)
%   minmax                 - function r = minmax(x, dim)
%   mod0                   - function y = mod0(x,b)
%   nans                   - |function out = nans(varargin)
%   ncol                   - function n = ncol(x)
%   ndgrid_jf              - function out = ndgrid_jf('cell', varargin)
%   nrms                   - function n = nrms(x, xtrue, arg)
%   nrow                   - function n = nrow(x)
%   num2list               - function varargout = num2list(x)
%   os_run                 - |function out = os_run(str)
%   outer_sum              - |function ss = outer_sum(xx,yy)
%   padn                   - |function out = padn(mat, newdim)
%   path_find_dir          - function fulldir = path_find_dir(part)
%   poisson                - function data = poisson(xm, seed, [options])
%   poisson0               - function data = poisson0(xm)
%   poisson1               - function data = poisson1(xmean, seed)
%   poisson2               - function data = poisson2(xm, [options])
%   poisson2unif           - function [uu] = poisson2unif(count, lambda)
%   pr                     - function pr(command)
%   printf                 - function printf(str, varargin)
%   printm                 - function printm(varargin)
%   printv                 - function printv(arg)
%   prompt                 - function out = prompt(arg)
%   rad2deg                - function d = rad2deg(r)
%   reale                  - |
%   rect                   - function y = rect(x)
%   remove_spaces          - |function arg = remove_spaces(arg)
%   repeat_slice           - function y = repeat_slice(x, n)
%   repout                 - function varargout = repout(in)
%   reshapee               - |function y = reshapee(x, varargin)
%   reshaper               - |function y = reshaper(x, dim)
%   rms                    - function [rh, sh] = rms(x)
%   run_mfile_local        - function run_mfile_local(arg)
%   sinc_periodic          - |function x = sinc_periodic(t, K)
%   single_ws              - single_ws
%   sino_mash              - function sino = sino_mash(sino, nr, nv, orbit)
%   spdiag                 - function b = spdiag(a, options)
%   stackpick              - function y = stackpick(x, ii)
%   stackup                - |function ss = stackup(x1, x2, ...)
%   streq                  - |function tf = streq(a, b [,n])
%   strreps                - function s = strreps(s, f1, r1, f2, r2, ...)
%   strum_test             - |function strum_test
%   subv2ind               - function index = subv2ind(dim, sub)
%   swapdim                - function y = swapdim(x, dim1, dim2)
%   test_all               - test_all.m
%   test_all_mex           - test_all_mex
%   test_all_util          - test_all_util.m
%   test_dir               - function tdir = test_dir(tdir)
%   ticker                 - |function ticker(varargin)
%   truncate_precision     - function y = truncate_precision(x, digits, [option])
%   unpadn                 - |function out = unpadn(mat, newdim)
%   upsample_rep           - function y = upsample_rep(x, m)
%   vararg_pair            - function [opt, extra] = vararg_pair(opt, varargs, [options])
%   vcorrcoef              - function c = vcorrcoef(u, v)
%   warn                   - function warn(varargin)
%   zero_tiny_negative     - function y = zero_tiny_negative(x, tol)
