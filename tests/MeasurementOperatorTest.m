classdef MeasurementOperatorTest < matlab.unittest.TestCase

    properties (SetAccess = private)
        N  % size(x)
        seed  % random seed
    end

    methods (TestClassSetup)

        function addToPath(testCase)
            p = path;
            testCase.addTeardown(@path, p);
            addpath('../nufft');
            addpath('../lib/operators');
            addpath('../data');
        end

        function setupProperties(testCase)
            testCase.N = [256, 512];
            testCase.seed = 1234;
        end

    end

    methods (Test)

        function testAdjointOpnufft(testCase)
            J = [7, 7];  % kernel size
            K = 2 * testCase.N;  % Fourier size
            nshift = testCase.N / 2;  % shift

            % coverage parameters
            T = 100;
            cov_type = 'vlaa';
            dl = 2.;
            hrs = 5;

            % generate test coverage
            [u_ab, v_ab, na] = generate_uv_coverage(T, hrs, dl, cov_type);
            om = [v_ab(:), u_ab(:)];
            M = na * (na - 1) / 2 * T;  % = numel(om(:, 1))

            % generate operators (forward and adjoint)
            [A, At, G, ~] = op_nufft(om, testCase.N, J, K, nshift);

            rng(testCase.seed);
            x = randn(testCase.N);
            y = (randn(M, 1) + 1i * randn(M, 1)) / sqrt(2);

            p1 = sum(conj(G * A(x)) .* y);
            p2 = sum(sum(conj(x) .* At(G' * y)));
            testCase.verifyTrue(abs(p1 - p2) / abs(p1) < 1e-10);
        end

        function testAdjointOppnufft(testCase)
            % TODO: test with more than a single block

            J = [7, 7];  % kernel size
            K = 2 * testCase.N;  % Fourier size
            nshift = testCase.N / 2;  % shift

            % coverage parameters
            T = 100;
            cov_type = 'vlaa';
            dl = 2.;
            hrs = 5;

            % generate test coverage
            [u_ab, v_ab, na] = generate_uv_coverage(T, hrs, dl, cov_type);
            om = [v_ab(:), u_ab(:)];
            M = T * na * (na - 1) / 2;  % = numel(om(:, 1))

            % generate operators (forward and adjoint)
            [A, At, G, W] = op_p_nufft([{om(:, 1)}, {om(:, 2)}], ...
                                          testCase.N, J, K, nshift);

            rng(testCase.seed);
            x = randn(testCase.N);
            y = (randn(M, 1) + 1i * randn(M, 1)) / sqrt(2);

            Ax = A(x);
            Ax = G{1} * Ax(W{1});
            Aty = zeros(prod(K), 1);
            Aty(W{1}) = G{1}' * y;
            Aty = At(Aty);

            p1 = sum(conj(Ax) .* y);
            p2 = sum(conj(x(:)) .* Aty(:));
            testCase.verifyTrue(abs(p1 - p2) / abs(p1) < 1e-10);
        end

    end
end
