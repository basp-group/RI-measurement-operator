import pynufft
import torch
import numpy as np
from pysrc.pynufft_backend import NUFFT_torch

class operator_pynufft:
    """ Radio-interferometric (RI) operator class with the associating forward and adjoint operators.
    Backprojection to create dirty image and residual generation are also included.
    
    """
    def __init__(self, im_size: tuple, device: torch.device = torch.device('cpu')):
        """Initializes the RI operator class with the image size and device

        Parameters
        ----------
        im_size : tuple
            (H, W) of the image of interest
        device : torch.device, optional
            Determine to work on cpu or gpu, by default torch.device('cpu')
        Raises
        ------
        AssertionError
            im_size should be of of length 2 (H, W)
        """
        assert len(im_size) == 2, "im_size should be of HxW"
        self.im_size = im_size
        self.device = device
        self.grid_size = (self.im_size[0]*2, self.im_size[1]*2)
        self.torchdevice = device
        if 'cpu' in str(device):
            self.device = 'cpu'
            print('Using CPU')
            self.NUFFT_obj = pynufft.NUFFT()
            self.fft2 = np.fft.fft2
            self.ifft2 = np.fft.ifft2
        elif 'cuda' in str(device):
            self.device = 'gpu'
            self.NUFFT_obj = NUFFT_torch(device)
            self.fft2 = torch.fft.fft2
            self.ifft2 = torch.fft.ifft2
        self.uv = None
        self.imweight = None
        self.PSF_peak_val = None
        self.weighting_on = False
        
    def validate_exact(self):
        assert self.uv is not None and self.imweight is not None, 'uv and imweight must be provided for exact measurement operator'
        
    def set_uv_imweight(self, uv : np.ndarray, imweight):
        """Set the Fourier sampling pattern and weighting to be applied to the measurement. If the operator type is
        'sparse_matrix', the interpolation matrix will be computed.
        
        Parameters
        ----------
        uv : numpy.ndarray
            Fourier sampling pattern, shape (B, 2, N)
        imweight : torch.Tensor or None
            Weighting to be applied to the measurement, shape (B, 1, N)
        """
        self.uv = uv
        if imweight is None:
            if 'cuda' in str(self.torchdevice):
                self.imweight = torch.ones(max(uv.shape), device=self.torchdevice)
            elif 'cpu' in str(self.torchdevice):
                self.imweight = np.ones(max(uv.shape))
        else:
            self.imweight = imweight
            self.weighting_on = True
        self.NUFFT_obj.plan(uv, self.im_size, self.grid_size, (7,7))

    def A(self, x, tau=0):
        """Forward operator.

        Parameters
        ----------
        x : torch.Tensor
            Image(s) for the forward operator to be applied on, shape (B, C, H, W)
        tau : float, optional
            Noise to be added to the measurement, by default 0

        Returns
        -------
        torch.Tensor
            Measurement associated to the image(s) of interest and forward operator, shape (B, 1, N)
        """
        y = self.NUFFT_obj.forward(x)
        if tau is not None:
            if tau > 0:
                y = self.noise(y, tau)
        return y * self.imweight
    
    def noise(self, y, tau):
        """Add noise to the measurement.

        Parameters
        ----------
        y : torch.Tensor
            Measurement to be added noise to, shape (B, 1, N)
        tau : float
            Standard deviation of the random Gaussian noise to be added

        Returns
        -------
        torch.Tensor
            Noisy measurement, shape (B, 1, N)
        """
        match self.device:
            case 'gpu':
                return y + (torch.randn_like(y) + 1j * torch.randn_like(y)) * tau / torch.sqrt(torch.tensor(2.))
            case 'cpu':
                return y + (np.random.randn(y.shape[0], y.shape[1]) + 1j * np.random.randn(y.shape[0], y.shape[1])) * tau / np.sqrt(2.)
    
    def At(self, y):
        """Adjoint operator.

        Parameters
        ----------
        y : torch.Tensor
            Measurement to be applied the adjoint operator on, shape (B, 1, N)

        Returns
        -------
        torch.Tensor
            Resulting image(s) from the adjoint operator, shape (B, C, H, W)
        """
        match self.device:
            case 'gpu':
                return torch.real(self.NUFFT_obj.adjoint(y * self.imweight))
            case 'cpu':
                return np.real(self.NUFFT_obj.adjoint(y * self.imweight)) * np.prod(self.grid_size)
    
    def backproj(self, x, tau=None):
        """Backprojection to create dirty image from the given image, using either exact measurement operator or
        approximation by convolution with PSF.

        Parameters
        ----------
        x : torch.Tensor
            Image of interest, shape (B, C, H, W).
        PSF : torch.Tensor, optional
            PSF of the measurement operator, shape (B, C, H, W). Required if op_acc is 'approx'
        tau : float, optional
            Standard deviation of a Gaussian distribution of noise to be added to the measurement, by default None

        Returns
        -------
        torch.Tensor
            Dirty image.
        """
        self.validate_exact()
        dirty = self.At(self.A(x, tau))
        if self.PSF_peak_val is not None:
            dirty /= self.PSF_peak_val
        else:
            dirty /= self.PSF_peak()
        return dirty
    
    def backproj_PSF(self, x, PSF):
        assert PSF is not None, 'PSF must be provided for approximated measurement operator'
        x_fft = self.fft2(x, s=self.grid_size, dim=[-2, -1])
        if PSF.size(-1) // self.im_size[-1] == 2:
            PSF_fft = self.fft2(PSF, dim=[-2, -1])
        elif PSF.size(-1) // self.im_size[-1] == 1:
            PSF_fft = self.fft2(PSF, s=self.grid_size, dim=[-2, -1])
        x_dirty = self.ifft2(x_fft * PSF_fft, dim=[-2, -1])
        match self.device:
            case 'gpu':
                return torch.real(x_dirty)
            case 'cpu':
                return np.real(x_dirty)
    
    def backproj_data(self, y):
        """Backprojection to create dirty image from the given image, using either exact measurement operator or
        approximation by convolution with PSF.

        Parameters
        ----------
        x : torch.Tensor
            Image of interest, shape (B, C, H, W).
        PSF : torch.Tensor, optional
            PSF of the measurement operator, shape (B, C, H, W). Required if op_acc is 'approx'
        tau : float, optional
            Standard deviation of a Gaussian distribution of noise to be added to the measurement, by default None

        Returns
        -------
        torch.Tensor
            Dirty image.
        """
        self.validate_exact()
        dirty = self.At(y)
        if self.PSF_peak_val is not None:
            dirty /= self.PSF_peak_val
        else:
            dirty /= self.PSF_peak()
        return dirty
        
    def gen_res(self, dirty, x):
        """Compute the residual dirty image for given dirty image and reconstruction image, using
        either exact measurement operator or approximation by convolution with PSF.

        Parameters
        ----------
        dirty : torch.Tensor
            Dirty image, shape (B, C, H, W).
        x : torch.Tensor
            Reconstruction image, shape (B, C, H, W).
        PSF : torch.Tensor, optional
            PSF of the measurement operator, shape (B, C, H, W). Required if op_acc is 'approx'

        Returns
        -------
        torch.Tensor
            Residual dirty image.
        """
        self.validate_exact()
        return dirty - self.backproj(x)
        
    def gen_res_PSF(self, dirty, x, PSF):
        if PSF.size(-1) // self.im_size[-1] == 2:
            start, end = x.size(3), x.size(3)*2
        elif PSF.size(-1) // self.im_size[-1] == 1:
            start, end = int(self.im_size[0]/2), int(self.im_size[0]+self.im_size[0]/2)
        return dirty - self.backproj_PSF(x, PSF=PSF)[..., start:end, start:end]
        
    def gen_PSF(self, batch_size=1, normalize=False):
        """Generate the point spread function (PSF) for the given Fourier sampling pattern and weighting.

        Parameters
        ----------
        os: int, optional
            Oversampling factor, by default 1.
        batch_size : int, optional
            Number of images in the batch, by default 1.

        Returns
        -------
        torch.Tensor
            PSF for the given Fourier sampling pattern and weighting.
        """
        self.validate_exact()
        # if os > 1:
        #     dirac_size = (self.im_size[0]*os, self.im_size[1]*os)
        #     op_PSF = operator(im_size=dirac_size, device=self.device, op_acc='exact')
        #     A_PSF = op_PSF.A
        #     At_PSF = op_PSF.At
        # else:
        #     dirac_size = self.im_size
        #     A_PSF = self.A
        #     At_PSF = self.At
        match self.device:
            case 'gpu':
                dirac_delta = torch.zeros(self.im_size).to(self.torchdevice)
                while len(dirac_delta.shape) < 4:
                    dirac_delta = dirac_delta.unsqueeze(0)
                if batch_size > 1:
                    assert self.uv.shape[0] == batch_size, "uv should have the same batch size as batch_size (in first dimension)"
                    dirac_delta = torch.stack([dirac_delta]*batch_size, dim=0)
                dirac_delta[..., self.im_size[0]//2, self.im_size[1]//2] = 1.
            case 'cpu':
                dirac_delta = np.zeros(self.im_size)
                dirac_delta[self.im_size[0]//2, self.im_size[1]//2] = 1.
        PSF = self.At(self.A(dirac_delta))
        if normalize and self.device == 'gpu':
            return PSF / torch.amax(PSF, dim=(-1, -2), keepdim=True)
        elif normalize and self.device == 'cpu':
            return PSF / np.amax(PSF, axis=(-1, -2))
        else:
            return PSF
    
    def PSF_peak(self, batch_size=1):
        """Compute the peak of PSF.

        Parameters
        ----------
        batch_size : int, optional
            Number of images in the batch, by default 1.

        Returns
        -------
        torch.Tensor
            A tensor containing the peak of PSF for the batch.
        """
        PSF = self.gen_PSF(batch_size)
        if self.device == 'gpu':
            PSF_peak = torch.amax(PSF, dim=(-1, -2), keepdim=True)
        elif self.device == 'cpu':
            PSF_peak = np.max(PSF, axis=(-1, -2))
        if self.PSF_peak_val is None:
            self.PSF_peak_val = PSF_peak
        return PSF_peak
    
    def op_norm(self, tol=1e-4, max_iter=500, verbose=0):
        val1 = self.op_norm_cal(tol=tol, max_iter=max_iter, verbose=verbose)
        if self.weighting_on:
            val2 = self.op_norm2(tol=tol, max_iter=max_iter, verbose=verbose)
            match self.device:
                case 'gpu':
                    eta_correction = torch.sqrt(val2 / val1)
                case 'cpu':
                    eta_correction = np.sqrt(val2 / val1)
        else:
            eta_correction = 1
        return val1, eta_correction
        

    def op_norm_cal(self, tol=1e-4, max_iter=500, verbose=0):
        """Compute spectral norm of the operator.

        Parameters
        ----------
        tol : float, optional
            Tolerance for relative difference on current and previous solution for stopping the algorithm, by default 1e-5.
        max_iter : int, optional
            Maximum number of iteration to compute, by default 500.
        verbose : int, optional
            By default 0.

        Returns
        -------
        float
            The computed spectral norm of the operator.
        """
        self.validate_exact()
        match self.device:
            case 'gpu':
                x = torch.randn(self.im_size).unsqueeze(0).unsqueeze(0).to(self.torchdevice)
                norm_fn = torch.linalg.norm
                abs_fn = torch.abs
            case 'cpu':
                x = np.random.randn(*self.im_size)
                norm_fn = np.linalg.norm
                abs_fn = np.abs
        x /= norm_fn(x)
        init_val = 1
        for k in range(max_iter):
            x = self.At(self.A(x))
            val = norm_fn(x)
            rel_var = abs_fn(val - init_val) / init_val
            if verbose > 1:
                print(f'Iter = {k}, norm = {val}')
            if rel_var < max(2e-6, tol):
                break
            init_val = val
            x = x / val
        if verbose > 0:
            print(f'Norm = {val}\n')
            
        return val
        
    def op_norm2(self, tol=1e-4, max_iter=500, verbose=0):
        imweight_tmp = self.imweight
        self.imweight = imweight_tmp**2
        val2 = self.op_norm_cal(tol=tol, max_iter=max_iter, verbose=verbose)
        self.imweight = imweight_tmp
        return val2