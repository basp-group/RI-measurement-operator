# RI-measurement-operator

![language](https://img.shields.io/badge/language-Python-orange.svg)
[![license](https://img.shields.io/badge/license-GPL--3.0-brightgreen.svg)](LICENSE)
<!-- [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit) -->

## Description

This branch provides a Python implementation of the RI measurement operator. The current implementation does not accommodate `w`-correction. Python version 3.10 or higher is required.

Python-based utility scripts to generate realistic Fourier sampling patterns and extract RI data from a measurement set (MS) to `.mat` file are available in  `pyutils/`. Instructions are provided in `pyutils/Readme.md`.


**Contributors**: by alphabetical order, T. Chu, A. Dabbech .


## Installation

Clone the current repository

```bash
git clone -b python  https://github.com/basp-group/RI-measurement-operator.git
```

## Dependencies 

The code relies on external NUFFT Python libraries, both implementing the NUFFT proposed by

> J. A. Fessler and B. P. Sutton, Nonuniform Fast Fourier Transforms Using Min-Max Interpolation, *IEEE Trans. Image Process.*, 51(2), 560-574, 2003.


- TorchKbNufft   [(https://github.com/mmuckley/torchkbnufft)](https://github.com/mmuckley/torchkbnufft);

- pyNUFFT [(https://github.com/jyhmiinlin/pynufft)](https://github.com/jyhmiinlin/pynufft).

### Pip
With the desired virtual environment activated, install the packages using the command below:
``` bash
pip install -r requirements.txt
```

### Conda
Create a new conda environment and install the packages from the provided `requirements_conda.yml` file:
```
conda env create -f requirements_conda.yml 
```
If the conda command is not recognized and for more details regarding conda, read [conda_install.md](conda_install.md).

## Examples

Two examples of usage are provided: 

1. `example_sim_measop.py` enables the simulation of the measurement operator and its adjoint from a Fourier sampling pattern.
   The script takes an input `.mat` file containing the `u`, `v`, `w` coordinates (in meter) and `frequency` (in MHz).  

   From the terminal, run the command below:
   ``` Python
   python  example_sim_measop.py --yaml_file config/sim_measop.yaml
   ```
   The script will load the yaml file `config/sim_measop.yaml` with the following arguments:
   ``` yaml
      data_file         # (str) Path to the file containing u, v, w, frequency, and imweight (optional)
      im_size_x         # (int) Target image dimension in the x direction
      im_size_y         # (int) Target image dimension in the y direction
      superresolution   # (float) Superrresolution facor, inferring the bandwidth of the imaged spatial Fourier domain
      nufft             # (str) Nufft library to be used, choices are ['pynufft', 'tkbn']
      on_gpu            # (bool) Run on GPU
   ```


2. `example_sim_ri_data.py` enables the simulation of radio-inteferometric data from a given ground truth image and a Fourier sampling pattern.

   From the terminal, run the command below:
   ``` Python
   python  example_sim_ri_data.py --yaml_file config/sim_ri_data.yaml
   ```
   The script will load the yaml file `config/sim_ri_data.yaml` with the following arguments:
   ``` yaml
   data_file            # (str) Path to the file containing u, v, w, frequency, and imweight (optional)
   gdth_file            # (str) Path to the ground truth file  (.fits)
   superresolution      # (float) Superrresolution facor, inferring the bandwidth of the imaged spatial Fourier domain
   nufft                # (str) Nufft library to be used, choices are ['pynufft', 'tkbn']
   noise_heuristic      # (float) Target dynamic range of the ground truth image used to infer the noise level (option 1)
   noise_isnr           # (float) Input signa-to-noise ratio to infer the noise level (option 2)
   on_gpu               # (bool) run on GPU
   dict_save_foldername # (str) Path to the folder where the dictionary will be saved
    ```
