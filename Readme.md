# RI-measurement-operator

![language](https://img.shields.io/badge/language-MATLAB-orange.svg)
[![license](https://img.shields.io/badge/license-GPL--3.0-brightgreen.svg)](LICENSE)
[![docs-page](https://img.shields.io/badge/docs-latest-blue)](https://basp-group.github.io/RI-measurement-operator/)
<!-- [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit) -->

## Description

``RI-measurement-operator`` is a radio-interferometry MATLAB library devoted to
the implementation of the measurement operator. The proposed implementation can accommodate:

- data-weighting schemes: Briggs and uniform. 
- `w`-correction VIA `w`-projection [(Dabbech2018)](https://academic.oup.com/mnras/article/476/3/2853/4855950);
- a compact Fourier model for the direction dependent effects (DDEs) [(Dabbech2021)](https://academic.oup.com/mnras/article-abstract/506/4/4855/6315336?redirectedFrom=fulltext);

Python-based utility scripts to generate realistic Fourier sampling patterns and extract RI data from a measurement set (MS) to `.mat` file are available in  `pyutils/`. Instructions are provided in `pyutils/Readme.md`.

The ``RI-measurement-operator`` library is a core dependency of the [`Faceted-Hyper-SARA`](https://github.com/basp-group/Faceted-Hyper-SARA) wideband imaging library for radio-interferometry, associated with the following publications.

>P.-A. Thouvenin, A. Abdulaziz, A. Dabbech, A. Repetti, Y. Wiaux, [Parallel faceted imaging in radio interferometry via proximal splitting (Faceted HyperSARA): I. Algorithm and simulations](https://academic.oup.com/mnras/article/521/1/1/6823714), *MNRAS*, 521(1), 1–19, 2023. 
>
>P.-A. Thouvenin, A. Dabbech, M. Jiang, J.-P. Thiran, A. Jackson, Y. Wiaux,
[Parallel faceted imaging in radio interferometry via proximal splitting (Faceted HyperSARA): II. Real data proof-of-concept and code](https://academic.oup.com/mnras/article/521/1/20/6823713?searchresult=1), *MNRAS*, 521(1), 20–34, 2023.

A rudimentary Python implementation of the measurement operator is available in the branch `python`.

**Contributors**: by alphabetical order, A. Dabbech, M. Jiang, A. Onose, P.-A. Thouvenin.


## Installation

Just clone the current repository

```bash
git clone https://github.com/basp-group/RI-measurement-operator.git
```

To get started with the library, take a look at the [documentation hosted online on github](https://basp-group.github.io/RI-measurement-operator/).

## Dependencies

- The present repository contains a slightly modified version of the MATLAB NUFFT algorithm available [online](http://web.eecs.umich.edu/~fessler/irt/fessler.tgz) (`irt/` folder), and described in

> J. A. Fessler and B. P. Sutton, Nonuniform Fast Fourier Transforms Using Min-Max Interpolation, *IEEE Trans. Image Process.*, 51(2), 560-574, 2003.

and also made available by the author on [github](https://github.com/JeffFessler/mirt). A lighter version of the non-uniform FFT available in `irt` (relying exclusively on the default parameters) is provided in `nufft/`.

- The repository also contains functions associated with the following publications

> A. Dabbech, L. Wolz, L. Pratley, J. D. McEwen and Y. Wiaux, [The w-effect in interferometric imaging: from a fast sparse measurement operator to superresolution](http://dx.doi.org/10.1093/mnras/stx1775), *MNRAS*, 471(4), 4300-4313, 2017.
>
> A. Onose, A. Dabbech and Y. Wiaux, [An accelerated splitting algorithm for radio-interferometric imaging: when natural and uniform weighting meet](http://dx.doi.org/10.1093/mnras/stx755), *MNRAS*, 469(1), 938-949, 2017.

## Examples

Two examples of usage are provided: 

1. `example_sim_measop.m` enables the simulation of the measurement operator and its adjoint from a Fourier sampling pattern.
   The script takes an input `.mat` file containing the `u`, `v`, `w` coordinates (in meter) and `frequency` (in MHz). Alternatively, a _toy_ Fourier sampling pattern can be generated using `data/` utilities.
2. `example_sim_ri_data.m` enables the simulation of radio-inteferometric data from a given ground truth image and a Fourier sampling pattern.
   

## Contributions

### Building the documentation

To build the documentation, make sure the following Python packages have been installed, and issue the appropriate buid command.

```bash
# setup conda environment to build the documentation
conda env create --name mo-doc --file environment.yml 

# alternative using conda/pip
# conda create -n mo-doc
# conda activate mo-doc
# conda install pip
# pip install -r requirement.txt

# building the documentation in html format
cd docs
make html
```

All the generated ``.html`` files are contained in the ``docs/build`` folder.

If needed, you can delete the `conda` environment as follows

```bash
conda env remove -n mo-doc
```

### Pushing the documentation online

Add a `worktree` from the `master` branch

```bash
# make sure the folder html does not exist before running the command
git worktree add docs/build/html gh-pages
cd docs/build/html
git add .
git commit -m "Build documentation as of $(git log '--format=format:%H' master -1)"
git push origin gh-pages
# delete the worktree
cd ../
git worktree remove html
```

### Code layout

Make sure any pull request has been properly formatted with the [`miss_hit`](https://pypi.org/project/miss-hit/) package using the `miss_hit.cfg` file provided

```bash
# activate mo-doc environment (see previous paragraph)
conda activate mo-doc
# run the following command from the root of the package (where the miss_hit.cfg file is)
mh_style --fix .
```
