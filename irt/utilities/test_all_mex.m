% test_all_mex
% make sure all mex files can execute
% by running the internal "check" of each.

list_spline = {
    'BsplCo2GdXMirr', ...
    'BsplCo2GdXTranMirr', ...
    'BsplCo2GdXTranZero', ...
    'BsplCo2GdXZero', ...
    'BsplCo2GdYMirr', ...
    'BsplCo2GdYTranMirr', ...
    'BsplCo2GdYTranZero', ...
    'BsplCo2GdYZero', ...
    'BsplCo2GdZMirr', ...
    'BsplCo2GdZTranMirr', ...
    'BsplCo2GdZTranZero', ...
    'BsplCo2GdZZero', ...
    'BsplCo2ValMirr', ...
    'BsplCo2ValTranMirr', ...
    'BsplCo2ValTranZero', ...
    'BsplCo2ValTranZeroFilt', ...
    'BsplCo2ValZero', ...
    'BsplCo2ValZeroFilt', ...
    'BsplExpand', ...
    'BsplReduce', ...
    'BsplVal2CoMirr', ...
    'BsplVal2CoZero'
};

list = {
    'jf_mex', ...
    'dtft_mex', ...
    'exp_xform_mex', ...
    'mri_exp_mult_mex', ...
...
    'interp1_table_adj_mex', ...
    'interp1_table_mex', ...
    'interp2_table_adj_mex', ...
    'interp2_table_mex', ...
    'interp3_table_adj_mex', ...
    'interp3_table_mex', ...
...
    'penalty_mex', ...
    'rotmex', ...
    'wtfmex', ...
    'f3d_mex', ...
...
    'delaysum1_mex', ...
    'ir_shrink1_mex', ...
    'ir_tridiag_inv_mex', ...
...
    list_spline{:}
};

% check for UM-only mex files
if exist('dd_ge1_mex') == 3
    list{end + 1} = 'dd_ge1_mex';
end
if exist('dd_ge2_mex') == 3
    list{end + 1} = 'dd_ge2_mex';
end

is_missing = false(1, numel(list));
for ii = 1:numel(list)
    mex = list{ii};
%   pr mex
    if exist(mex) ~= 3
        is_missing(ii) = true;
    end
end

if any(is_missing)
    printf(' ');
    printm('These mex files are missing:');
%   pr find(is_missing)
%   list{is_missing}
    pr list(is_missing);
prompt;
    list = list(~is_missing);
end

passed = '';
failed = '';
missing = '';
for ii = 1:numel(list)
    mex = list{ii};
    pr mex;
    if exist(mex) ~= 3
        missing = [missing ' ' mex];
        continue
    end
    try
        fun = str2func(mex);
        fun('check');
        passed = [passed ' ' mex];
    catch
        failed = [failed ' ' mex];
    end
end

if ~isempty(missing) || ~isempty(failed)

    if ~isempty(missing)
        printm(['\nThese mex files are missing: ' missing]);
    end

    if ~isempty(failed)
        printm(['\nThese mex files failed: ' failed]);
    end

    if ~isempty(passed)
        printm(['\nThese mex files passed: ' passed]);
        printm 'So perhaps some things will still work.';
    end

    printm 'Sorry, you seem to have mex problems. :-(';
    printm 'Probably you are a PC user and Windoze is not supported.';
    printm 'Or (in linux) there may be a gcc library version issue?';
    printm 'Or you may have a path problem.';

else
    printm '\n----------------------------------------------------------';
    printm(['All mex files present and passed:\n' passed]);
    printm '----------------------------------------------------------\n';
end
