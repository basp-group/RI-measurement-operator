% delaysum1_test.m
% Steve Schmitt

Nh = 10000;
Nx = 100;
Nm = 500;
Ny = Nh;

rng(0)
h = randn(Nh,1);
d = round(randn(Nx,Nm)*100);
x = randn(Nx,1);
nthread = jf('ncore');

tic;
yt = delaysum1_mex('delaysum1,forw,thr',...
	single(h),...
	single(d),...
	int32(nthread),...
	single(x),...
	int32(Ny),...
	int32(0));
t1 = toc;

tic;
y = delaysum1_mex('delaysum1,forw',...
	single(h),...
	single(d),...
	int32(1),...
	single(x),...
	int32(Ny), ...
	int32(0));
t2 = toc;

printm('forward:\n threaded time = %.3f\n non-threaded time = %.3f\n speedup = %.2f\n\n',t1,t2,t2/t1);

if any(y ~= yt)
	fail('THREADED != NON-THREADED')
end

tic;
b = delaysum1_mex('delaysum1,back',...
	single(h),...
	single(d),...
	int32(1),...
	single(y), ...
	int32(0));
t3 = toc;

tic;
bt = delaysum1_mex('delaysum1,back,thr',...
	single(h),...
	single(d),...
	int32(nthread),...
	single(y),...
	int32(0));
t4 = toc;

printm('back:\n threaded time = %.3f\n non-threaded time = %.3f\n speedup = %.2f for %d threads\n\n',t4,t3,t3/t4, nthread);

if any(b ~= bt)
	fail('THREADED != NON-THREADED')
end
