# Measurement Operator

This repository contains a collection of MATLAB functions to implement the measurement operator involved in radio-interferometry.

**Contributors**: A. Dabbech, A. Onose, P.-A. Thouvenin.

**Dependencies**: The present repository contains a slightly modified version of the MATLAB NUFFT algorithm available [online](http://web.eecs.umich.edu/~fessler/irt/fessler.tgz) (`irt/` folder), and described in
> J. A. Fessler and B. P. Sutton, Nonuniform Fast Fourier Transforms Using Min-Max Interpolation, *IEEE Trans. Image Process.*, vol. 51, n. 2, pp. 560--574, Feb. 2003.

and also made available by the author on [github](https://github.com/JeffFessler/mirt)). A lighter version of the non-uniform FFT available in `irt` (relying exclusively on the default parameters) is provided in `nufft/`.

It also contains additional functions associated with the following publications
> A. Dabbech, L. Wolz, L. Pratley, J. D. McEwen and Y. Wiaux, [The w-effect in interferometric imaging: from a fast sparse measurement operator to superresolution](http://dx.doi.org/10.1093/mnras/stx1775), *Mon. Not. Roy. Astron. Soc.*, 471(4):4300-4313, 2017.
>
> A. Onose, A. Dabbech and Y. Wiaux, [An accelerated splitting algorithm for radio-interferometric imaging: when natural and uniform weighting meet](http://dx.doi.org/10.1093/mnras/stx755), *Mon. Not. Roy. Astron. Soc.*, 469(1):938-949, 2017.
