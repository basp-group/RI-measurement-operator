# Requirements

1. Casacore https://casacore.github.io/casacore/
2. Meqtrees https://github.com/ratt-ru/meqtrees/wiki

# Utilities

## Fourier sampling pattern simulation
 
Simulation of realistic Fourier sampling patterns from antenna configurations of radio telescopes (`/pyutils/observatories`), performed using the utility script `sim_vla_ms.py`. The script relies heavily on [simms](https://github.com/ratt-ru/simms), that is part of the software package [Meqtrees](https://github.com/ratt-ru/meqtrees/wiki).

The task should be performed in its parent directory  `pyutils/`

Random variations of the observation setting using VLA antenna configurations are supported. These variations can be also extended to other radio telescopes.

The output files of the script are saved in three subdirectories:
   - `pyutils/vla_sims/ms/`: simulated _empty_ measurement sets. 
   - `pyutils/vla_sims/uvw/`: Fourier sampling patterns saved in  `.mat` file. The file encompasses the following fields:

``` matlab    
       "u"                     %  vector; `u`coordinate (in meter)
       "v"                     %  vector; `v` coordinate (in meter)
       "w"                     %  vector; `w` coordinate (in meter)
       "frequency"             %  scalar;  channel frequency (in MHz)
       "nominal_pixelsize"     %  scalar; maximum pixel size (in arcsec), corresponding to nominal resolution of the observations
```    
   - `pyutils/vla_sims/png/`: plots of the simulated uv-coverages  saved in `.png` file (for information only).

 The script  `sim_vla_ms.py` is an example. The user is encouraged to tailor if needed.

### Example
Multiple sampling patterns can be generated at once using the argument `-n`.

From the terminal launch:
``` python
    python sim_vla_ms.py -n 1 
```
## Extraction of RI data 

Data extraction from a measurement set, performed using the utility script `pyxis_ms2mat.py`.
The script relies heavily on [pyxis](https://github.com/ratt-ru/pyxis), that is part of the software package [Meqtrees](https://github.com/ratt-ru/meqtrees/wiki).

The task should be performed in its parent directory  `pyutils/`.
 
Extracted `.mat` file is saved in `pyutils/data/`. The file encompasses the following fields:

``` matlab
   "frequency"        %  scalar;  channel frequency
   "y"                %  complex vector; data (Stokes I)
   "u"                %  vector; `u`coordinate (in units of the wavelength)
   "v"                %  vector; `v` coordinate (in units of the wavelength)
   "w"                %  vector; `w` coordinate (in units of the wavelength)
   "nW"               %  vector;  inverse of the noise standard deviation 
   "nWimag"           %  (optional) vector; square root of the imaging weights if available (Briggs or uniform), empty otherwise
   "maxProjBaseline"  %  scalar; maximum projected baseline (in units of the wavelength)
  ```    

 The script  `pyxis_ms2mat.py` is an example. The user is encouraged to tailor it if the measurement set is organised differently.

### Example
Extracting (monochromatic) data at the frequency channel  `0` corresponding to the target source with field ID `0`.

The user must provide the name/path to the measurement set `$MS`. The following inputs are optional:
```python
    $SRCNAME  # default `SRCNAME=""`; source nametag
    $FIELDID  # default `FIELDID=0`; ID of the target source
    $FREQID   # default `FREQID=0`; ID of the channel to be extracted
```

From the terminal launch:
``` python
   pyxis  MS=$MS SRCNAME=3c353 FIELDID=0 FREQID=0 ms2mat
```

Data will be saved as .mat files in the sub-directory  `pyutils/data/`. The outcome is as follows:
``` bash
   pyutils/data/3c353_data_ch_1.mat
```
