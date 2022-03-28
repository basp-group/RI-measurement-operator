# Building the documentation from scratch

## Installation

Check documentation [here](https://github.com/sphinx-contrib/matlabdomain) for `sphinxcontrib-matlabdomain`.

```bash
conda install -c anaconda sphinx
conda install -c conda-forge sphinx_rtd_theme
conda install sphinxcontrib-napoleon --channel conda-forge
pip install -U sphinxcontrib-matlabdomain
sphinx-quickstart docs # select yes for separating build and source
# update conf.py with the proper elements
# building the documentation in html format
cd docs && make html
# sphinx-apidoc [OPTIONS] -o <OUTPUT_PATH> <MODULE_PATH> [EXCLUDE_PATTERN …]
cd source && sphinx-apidoc -f -o . ../..
```
