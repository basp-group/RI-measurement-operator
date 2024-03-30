# RI-measurement-operator

![language](https://img.shields.io/badge/language-Python-orange.svg)
[![license](https://img.shields.io/badge/license-GPL--3.0-brightgreen.svg)](LICENSE)
<!-- [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit) -->

## Description

This branch provides a Python implementation of the RI measurement operator. The current implementation does not accommodate `w`-correction.

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


The user has the possibility to select the NUFFT library of his choice.

Install the packages using the command below:
``` bash
   pip install -r requirement.txt
```
## Examples

Two examples of usage are provided: 

1. `example_sim_measop.py` enables the simulation of the measurement operator and its adjoint from a Fourier sampling pattern.
   The script takes an input `.mat` file containing the `u`, `v`, `w` coordinates (in meter) and `frequency` (in MHz).  

   From the terminal, run the command below:
   ``` Python
   python3  example_sim_measop.py \
   --data_file "data/uvw.mat"     \ # Path to the file containing u, v, w, frequency, and imweight (optional)
   --im_size 512 512              \ # target image dimension
   --superresolution 1.5          \ # Superrresolution facor, inferring the bandwidth of the imaged spatial Fourier domain
   --nufft 'pynufft'              \ # Nufft library to be used, choices are ['pynufft', 'tkbn']
   --on_gpu                         # run on GPU
   ```



2. `example_sim_ri_data.pu` enables the simulation of radio-inteferometric data from a given ground truth image and a Fourier sampling pattern.

   From the terminal, run the command below:
   ``` Python
   python3  example_sim_ri_data.py \
   --data_file  "data/3c353/3c353_data.mat"  \ # Path to the file containing u, v, w, frequency, and imweight (optional)
   --gdth_file  "data/3c353/3c353.fits"      \ # Path to the ground truth file  (.fits)
   --superresolution 1.5                     \ # Superrresolution facor, inferring the bandwidth of the imaged spatial Fourier domain
   --nufft 'pynufft'                         \ # Nufft library to be used, choices are ['pynufft', 'tkbn']
   --noise_heuristic 1e5                     \ # Target dynamic range of the ground truth image used to infer the noise level (option 1)
   --noise_isnr                              \ # Input signa-to-noise ratio to infer the noise level (option 2)
   --on_gpu                                    # run on GPU
    ```
