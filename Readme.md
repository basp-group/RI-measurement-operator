# Measurement Operator

This repository contains a collection of MATLAB functions to implement the measurement operator involved in radio-interferometry.

**Contributors**: A. Dabbech, A. Onose, P.-A. Thouvenin.

**Dependencies**: The present repository contains a slightly modified version of the MATLAB NUFFT algorithm available at http://web.eecs.umich.edu/~fessler/irt/fessler.tgz (`irt/` folder), described in
> J. A. Fessler and B. P. Sutton - <strong>Nonuniform Fast Fourier Transforms Using Min-Max Interpolation</strong>, <em>IEEE Trans. Image Process.</em>, vol. 51, n. 2, pp. 560--574, Feb. 2003.
and also made available by the author on [github](https://github.com/JeffFessler/mirt)).

A lighter version of the non-uniform FFT available in `irt` (relying exclusively on the default parameters) is provided in `nufft/`.
