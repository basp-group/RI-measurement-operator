function val = op_norm(A, At, im_size, tol, max_iter, verbose)
% computes the maximum eigen value of the compund 
% operator AtA
%%
%A. Onose, A. Dabbech, Y. Wiaux - An accelerated splitting algorithm for radio-interferometric %imaging: when natural and uniform weighting meet, MNRAS 2017, arXiv:1701.01748
%https://github.com/basp-group/SARA-PPD
%%
x = randn(im_size);
x = x/norm(x(:));
init_val = 1;

for k = 1:max_iter
    y = A(x);
    x = At(y);
    val = norm(x(:));
    rel_var = abs(val-init_val)/init_val;
    if (verbose > 1)
        fprintf('Iter = %i, norm = %e \n',k,val);
    end
    if (rel_var < tol)
       break;
    end
    init_val = val;
    x = x/val;
    
end

if (verbose > 0)
    fprintf('Norm = %e \n\n', val);
end

end

