import argparse
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.io import savemat, loadmat

from example_sim_measop import gen_measop

def parse_args():
    parser = argparse.ArgumentParser(description='Read uv and imweight files')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to the file containing u, v and imweight')
    parser.add_argument('--gdth_file', type=str, required=True,
                        help='Path to the file containing the ground truth image')
    parser.add_argument('--superresolution', type=float, default=1.,
                    help='Super resolution factor')
    parser.add_argument('--nufft', choices=['pynufft', 'tkbn'], required=True,
                        help='Nufft library to be used')
    parser.add_argument('--noise_heuristic', type=float, default=None,
                        help='Target dynamic range of the ground truth image')
    parser.add_argument('--noise_isnr', type=float, default=None,
                        help='Input signa-to-noise ratio')
    parser.add_argument('--on_gpu', action='store_true',
                        help='Utilise GPU')
    parser.add_argument('--dict_save_path', type=str, default='./results/',
                        help='Path to save the dictionary containing data')
    return parser.parse_args()
        
def main(args):
    if not os.path.exists(args.dict_save_path):
        os.makedirs(args.dict_save_path)
    args.fname = args.gdth_file.split('/')[-1].split('.')[0]
    # Example script to simulate RI data
    # ground truth image
    gdthim = fits.getdata(args.gdth_file)
    gdthim /= gdthim.max()
    plt.figure(figsize=(10, 10))
    # display the normalized ground truth image with peak pixel intensity = 1
    plt.imshow(gdthim.squeeze(), cmap='afmhot')
    plt.colorbar()
    plt.title('Ground truth image')
    args.im_size = gdthim.shape
    
    # create measurement operator object from specified arguments
    # measurement operator = measop.A
    # adjoint measurement operator = measop.At
    measop, nWimag = gen_measop(args)
    
    if args.on_gpu or args.nufft == 'tkbn':
        gdthim = torch.tensor(gdthim.astype(np.float32))
    if args.nufft == 'tkbn':
        gdthim = gdthim.unsqueeze(0).unsqueeze(0)

    # model clean visibilities
    print('Generate model visibilities .. ')
    vis = measop.A(gdthim)
    # number of data points
    nmeas = np.max(vis.shape)
    
    # noise vector
    if args.noise_heuristic is not None:
        print('Generate noise (noise level commensurate of the target dynamic range)...')
        targetDynamicRange = args.noise_heuristic
        # compute measop spectral norm to infer the noise heuristic
        # eta_correction = 1 if nWimag is not in data file
        measop.set_uv_imweight(measop.uv, nWimag)
        measopSpectralNorm, eta_correction = measop.op_norm(tol=1e-4, max_iter=500)
        measop.set_uv_imweight(measop.uv, None)
        # noise standard deviation heuristic
        tau = np.sqrt(2 * measopSpectralNorm) / targetDynamicRange / eta_correction
        # noise realization(mean-0; std-tau)
        noise = tau * (np.random.randn(nmeas) + 1j * np.random.randn(nmeas)) / np.sqrt(2)
        # input signal to noise ratio
        isnr = 20 * np.log10(np.linalg.norm(vis) / np.linalg.norm(noise))
        print(f'INFO: random Gaussian noise with input SNR: {isnr:.3f} dB')
    elif args.noise_isnr is not None:
        print('generate noise from input SNR...')
        isnr = args.noise_isnr
        # user-specified input signal to noise ratio
        tau = np.linalg.norm(vis) / (10**(isnr/20)) / np.sqrt( (nmeas + 2 * np.sqrt(nmeas)))
        noise = tau * (np.random.randn(nmeas) + 1j * np.random.randn(nmeas)) / np.sqrt(2)
            
    # data
    print('Simulate data...')
    y = vis + noise
    
    # back-projected data
    print('Get back-projected data...')
    dirty = measop.At(y * nWimag**2)
    
    # display the non-normalized dirty image
    plt.figure(figsize=(10, 10))
    plt.imshow(dirty.squeeze(), cmap='afmhot')
    plt.colorbar()
    extra_str = '(weights applied)'
    plt.title(f'(Non-normalised) dirty image {extra_str if measop.weighting_on else ""}')
    
    print('Done')
    
    data_dict = loadmat(f'{args.dict_save_path}/{args.fname}_data.mat')

    
    data_dict.update({'y': y.numpy(force=True) if 'torch' in str(type(y)) else y,
                      'nW': tau * np.ones((nmeas,1))})
                      
     #reshape vars
    data_dict.update({'y':  np.reshape(y, (nmeas, 1))}   )
    savemat(f'{args.dict_save_path}/{args.fname}_data.mat', data_dict)
    
    # Compute RI normalization factor (just for info)
    ri_normalization = measop.PSF_peak()
    dirty_normalized = dirty / ri_normalization
    if 'torch' in str(type(dirty)):
        dirty_normalized = dirty_normalized.numpy(force=True)
    fits.writeto(f'{args.dict_save_path}/dirty_normalized.fits', dirty_normalized.squeeze(), overwrite=True)
    
if __name__ == '__main__':
    args = parse_args()
    args.device = torch.device('cuda') if args.on_gpu and torch.cuda.is_available() else torch.device('cpu')
    assert args.noise_heuristic is not None or args.noise_isnr is not None, 'Please provide noise heuristic or input SNR'
    if args.noise_heuristic is not None and args.noise_isnr is not None:
        print('Both noise heuristic and input SNR are provided. Using noise heuristic.')
    main(args)