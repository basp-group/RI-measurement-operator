function [proj] = solver_find_elipse_point(y, pU, A, T, xsol, v2, W, epsilont, max_iterations, min_iterations, eps, use_fb, max_no_el)
% Generate an initial point to initialize ``solver_proj_elipse_fb``.
%
% Parameters
% ----------
% y : cell{:} of complex[:]
%     Input data. Each cell entry corresponds to a data block.
% pU : cell{:} of double[:]
%     Diagonal preconditioner (encoded as a vector) involved in the
%     definition of the ellipsoid.
% A : anonymous function
%     Function handle representing the measurement operator.
% T : cell{:} of double[:]
%     Noise whitening diagonal matrices (encoded as vectors) associated
%     with each data block. Each cell entry corresponds to a data block.
% xsol : double[:, :]
%     Inital image estimate.
% v2 : cell{:} of complex[:]
%     Data-fidelity dual variables associated with each data block.
% W : cell{:} of bool[:]
%     Cell of boolean vector indicating the global position of each data
%     point in the full Fourier space (undo blocking).
% epsilont : cell{:} of double
%     Radius of the ellispoid.
% max_iterations : int
%     Maximum number of iterations.
% min_iterations : int
%     Minimum number of iterations.
% eps : double
%     Stopping criterion tolerance (relative variation in norm between two
%     consecutive iterates).
% use_fb : bool
%     Flag to use a forward-backward algorithm.
% max_no_el : int
%     Maximum number of elements per block (?).
%
% Returns
% -------
% proj : double[:]
%     Projection of the data `y` onto the ellipsoid considered.
%
% Note
% ----
% Deprecated function, still used in :cite:p:`Onose2017`,
% https://github.com/basp-group/SARA-PPD.
%%
if ~exist('use_fb', 'var')
    use_fb = 1;
end
if ~exist('max_no_el', 'var')
    max_no_el = 500;
end

R = size(y, 1);
proj = cell(R, 1);

if ~use_fb
    for q = 1:R
        nel = min(max_no_el, 2 * length(pU{q}));

        y_ = [real(y{q}); imag(y{q})];
        pU_ = 1 ./ [pU{q}; pU{q}];
        proj_ = 1 ./ sqrt(pU_) .* y_;
        proj_(1:nel) = 0;

        options = optimoptions(@fmincon, 'Algorithm', 'interior-point', ...
            'GradObj', 'on', 'GradConstr', 'on', 'TolX', 1e-6, 'Hessian', {'lbfgs', 5}, ...
            'Display', 'iter', 'Diagnostics', 'on');
        Q_ = pU_(1:nel);
        f_ = -sqrt(pU_(1:nel)) .* y_(1:nel);
        c_ = 0.5 * (y_' * y_) - 0.5 * epsilont{q}^2 - proj_(nel + 1:end)' * (sqrt(pU_(nel + 1:end)) .* y_(nel + 1:end)) ...
            + 0.5 * proj_(nel + 1:end)' * (pU_(nel + 1:end) .* proj_(nel + 1:end));

        fun = @(x)quadobj(x, Q_, f_, c_);

        H_ = pU_(1:nel);
        k_ = -sqrt(pU_(1:nel)) .* y_(1:nel);
        d_ = 0.5 * (y_' * y_) - 0.5 * epsilont{q}^2 - proj_(nel + 1:end)' * (sqrt(pU_(nel + 1:end)) .* y_(nel + 1:end)) ...
            + 0.5 * proj_(nel + 1:end)' * (pU_(nel + 1:end) .* proj_(nel + 1:end));

        nonlconstr = @(x)quadconstr(x, H_, k_, d_);

        [proj_(1:nel)] = fmincon(fun, proj_(1:nel), ...
            [], [], [], [], [], [], nonlconstr, options);

        proj{q} = proj_(1:end / 2) + 1i * proj_(end / 2 + 1:end);
    end
end

if use_fb
    % non gridded measurements of current solution
    ns = A(xsol);

    % partial non gridded measurements for each node
    ns_p = cell(R, 1);

    % select parts to be sent to nodes
    for q = 1:R
        ns_p{q} = ns(W{q});
    end

    for q = 1:R
        r2 = T{q} * ns_p{q};
        proj{q} = sqrt(pU{q}) .* solver_proj_elipse_fb(1 ./ pU{q} .* v2{q}, r2, y{q}, pU{q}, epsilont{q}, zeros(size(y{q})), max_iterations, min_iterations, eps);
    end
end

end

function [y, grady] = quadobj(x, Q, f, c)
    y = 1 / 2 * x' * (Q .* x) + f' * x + c;
    grady = (Q .* x) + f;
end

function [y, yeq, grady, gradyeq] = quadconstr(x, H, k, d)
    yeq = 0.5 * x' * (H .* x) + k' * x + d;
    y = [];

    gradyeq = (H .* x) + k;
    grady = [];
end
