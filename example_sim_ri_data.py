import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

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
    return parser.parse_args()
        
def main(args):
    # Example script to simulate RI data
    # ground truth image
    gdthim = fits.getdata(args.gdth_file)
    plt.figure(figsize=(10, 10))
    # display the normalized ground truth image with peak pixel intensity = 1
    plt.imshow(gdthim.squeeze()/ gdthim.max(), cmap='afmhot')
    plt.colorbar()
    plt.title('Ground truth image')
    args.im_size = gdthim.shape
    
    # create measurement operator object from specified arguments
    # measurement operator = measop.A
    # adjoint measurement operator = measop.At
    measop = gen_measop(args)
    
    if args.on_gpu or args.nufft == 'tkbn':
        gdthim = torch.tensor(gdthim.astype(np.float32))
    if args.nufft == 'tkbn':
        gdthim = gdthim.unsqueeze(0).unsqueeze(0)
    
    # model clean visibilities
    vis = measop.A(gdthim)
    # number of data points
    nmeas = np.max(vis.shape)
    
    # noise vector
    if args.noise_heuristic is not None:
        print('Generate noise (noise level commensurate of the target dynamic range)...')
        targetDynamicRange = args.noise_heuristic
        # compute measop spectral norm to infer the noise heuristic
        measopSpectralNorm = measop.op_norm(tol=1e-6, max_iter=500)
        # noise standard deviation heuristic
        tau = np.sqrt(2 * measopSpectralNorm) / targetDynamicRange
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
    dirty = measop.At(y)
    
    # display the non-normalized dirty image
    plt.figure(figsize=(10, 10))
    plt.imshow(dirty.squeeze(), cmap='afmhot')
    plt.colorbar()
    plt.title('(Non-normalised) dirty image')
    
    print('Done')
    
    # Compute RI normalization factor (just for info)
    # ri_normalization = measop.PSF_peak()
    
    
if __name__ == '__main__':
    args = parse_args()
    args.device = torch.device('cuda') if args.on_gpu and torch.cuda.is_available() else torch.device('cpu')
    assert args.noise_heuristic is not None or args.noise_isnr is not None, 'Please provide noise heuristic or input SNR'
    if args.noise_heuristic is not None and args.noise_isnr is not None:
        print('Both noise heuristic and input SNR are provided. Using noise heuristic.')
    main(args)