function [val, rel_var] = op_norm(A, At, im_size, tol, max_iter, verbose)
% Computes the maximum eigenvalue of the compound operator :math:`A^T A`.
%
% Parameters
% ----------
% A : function handle
%     Function handle for the direct operator.
% At : function handle
%     Function handle for the adjoint operator.
% im_size : int[2]
%     Size of the input image.
% tol : double
%     Tolerance for the stopping criterion (relative variation).
% max_iter : int
%     Maximum number of iterations.
% verbose : bool
%     Activate verbose mode.
%
% Returns
% -------
% val : double
%     Operator norm of :math:`A^TA` at convergence.
% rel_var : double
%     Value of the relative variation criterion at convergence.
%

%%
x = randn(im_size);
x = x / norm(x(:));
init_val = 1;

for k = 1:max_iter
    y = A(x);
    x = At(y);
    val = norm(x(:));
    rel_var = abs(val - init_val) / init_val;
    if verbose > 1
        fprintf('Iter = %i, norm = %e \n', k, val);
    end
    if rel_var < tol
       break
    end
    init_val = val;
    x = x / val;

end

if verbose > 0
    fprintf('Norm = %e \n\n', val);
end

end
