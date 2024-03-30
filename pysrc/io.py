import numpy as np
import torch
from astropy.io import fits
from scipy.constants import speed_of_light
from scipy.io import loadmat

def read_fits_as_tensor(path):
    x = fits.getdata(path)
    x = torch.tensor(x.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return x

def read_uv(uv_file_path, superresolution, device=torch.device('cpu'), nufft='pynufft'):
    """Read u, v and imweight from specified path.

    Parameters
    ----------
    uv_file_path : str
        Path to the file containing u, v and imweight.
    superresolution : float
        Super resolution factor.
    device : torch.device, optional
        Device to be used, by default torch.device('cpu')
    nufft : str
        Nufft library to be used, by default 'pynufft'

    Returns
    -------
    uv: torch.Tensor
        Fourier sampling pattern, shape (N, 2) or (1, 2, N)
    """
    uv_file = loadmat(uv_file_path)
    frequency = uv_file['frequency'].squeeze()
    # convert uvw in units of the wavelength
    umeter = uv_file['u'].squeeze() / (speed_of_light / frequency)
    vmeter = uv_file['v'].squeeze() / (speed_of_light / frequency)
    # wmeter = uv_file['w'].squeeze() / (speed_of_light / frequency)
    uv = np.stack((-vmeter, umeter), axis=1).astype(np.float32).squeeze()
    # maximum projected baseline (just for info)
    maxProjBaseline = np.max(np.sqrt(np.sum(uv**2, axis=np.argmin(uv.shape))))
    
    if 'nominal_pixelsize' in uv_file:
        pixelsize = uv_file['nominal_pixelsize'].squeeze()/superresolution
        superresolution = (180 / np.pi) * 3600 / (pixelsize * 2 * maxProjBaseline)
        print(f'INFO: user-specified pixel size: {pixelsize:.4e} arcsec (i.e. super resolution factor: {superresolution:.3f})')
    else:
        pixelsize = (180 / np.pi) * 3600 / (superresolution * 2 * maxProjBaseline)
        print(f'INFO: user-specified super resolution factor: {superresolution:.3f} (i.e. pixel size: {pixelsize:.4e} arcsec)')

    imaging_bandwidth = maxProjBaseline * superresolution
    uv = uv * np.pi / imaging_bandwidth
    
    if nufft == 'tkbn':
        uv = torch.tensor(uv).unsqueeze(0).to(device)
        if uv.size(1) > uv.size(2):
            uv = uv.permute(0, 2, 1)
    return uv