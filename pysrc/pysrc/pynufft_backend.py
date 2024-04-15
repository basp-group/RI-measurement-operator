import numpy
from importlib import import_module

class NUFFT_torch:
    """
    A tentative torch interface
    """
    
    
    def __init__(self, device):
        self.dtype = numpy.complex64  # : initial value: numpy.complex64
        self.debug = 0  #: initial value: 0
        self.Nd = ()  # : initial value: ()
        self.Kd = ()  # : initial value: ()
        self.Jd = ()  #: initial value: ()
        self.ndims = 0  # : initial value: 0
        self.ft_axes = ()  # : initial value: ()
        self.batch = None  # : initial value: None
        # self.processor='torch'
        self.device = device
        self.torch = import_module('torch')
        
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor.
        
           https://github.com/DSE-MSU/DeepRobust
        """
        sparse_mx = sparse_mx.tocoo().astype(numpy.complex64)
        indices = self.torch.from_numpy(
            numpy.vstack((sparse_mx.row, sparse_mx.col)).astype(numpy.int64))
        values = self.torch.from_numpy(sparse_mx.data)
        shape = self.torch.Size(sparse_mx.shape)
        return self.torch.sparse.FloatTensor(indices, values, shape) 

    def plan(self, om, Nd, Kd, Jd):
        """
        Plan the NUFFT object with the geometry provided.
    
        :param om: The M off-grid locates in the frequency domain,
                    which is normalized between [-pi, pi]
        :param Nd: The matrix size of the equispaced image.
                   Example: Nd=(256,256) for a 2D image;
                             Nd = (128,128,128) for a 3D image
        :param Kd: The matrix size of the oversampled frequency grid.
                   Example: Kd=(512,512) for 2D image;
                            Kd = (256,256,256) for a 3D image
        :param Jd: The interpolator size.
                   Example: Jd=(6,6) for 2D image;
                            Jd = (6,6,6) for a 3D image
        :param ft_axes: (Optional) The axes for Fourier transform.
                        The default is all axes if 'None' is given.
        :param batch: (Optional) Batch mode.
                     If the batch is provided, the last appended axis is the number
                     of identical NUFFT to be transformed.
                     The default is 'None'.
        :type om: numpy.float array, matrix size = M * ndims
        :type Nd: tuple, ndims integer elements.
        :type Kd: tuple, ndims integer elements.
        :type Jd: tuple, ndims integer elements.
        :type ft_axes: None, or tuple with optional integer elements.
        :returns: 0
        :rtype: int, float
    
        :ivar Nd: initial value: Nd
        :ivar Kd: initial value: Kd
        :ivar Jd: initial value: Jd
        :ivar ft_axes: initial value: None
    
        :Example:
    
        >>> from pynufft import NUFFT
        >>> NufftObj = NUFFT()
        >>> NufftObj.plan(om, Nd, Kd, Jd)
    
        or
    
        >>> NufftObj.plan(om, Nd, Kd, Jd, ft_axes)
    
        """
        from pynufft.src._helper import helper#, helper1
        # from ..src._helper import helper#, helper1
        self.ndims = len(Nd)  # : initial value: len(Nd)
        ft_axes = tuple(jj for jj in range(0, self.ndims))
        self.st = helper.plan(om, Nd, Kd, Jd, ft_axes=ft_axes,
                              format='CSR')
    
        self.Nd = self.st['Nd']  # backup
        self.Kd = self.st['Kd']
        self.sn = self.torch.from_numpy(numpy.asarray(self.st['sn'].astype(self.dtype), order='C')).to(self.device)
        G = self.st['p']
        self.mask_G = numpy.array((numpy.sum(numpy.abs(G), axis=0))).squeeze().astype('bool')
        Gm = G[:, self.mask_G]
        Gmt = Gm.conj().T
        self.torch_sp = self.sparse_mx_to_torch_sparse_tensor(Gm).to(self.device)
        self.torch_spH = self.sparse_mx_to_torch_sparse_tensor(Gmt).to(self.device)
    
        return 0
    
    def forward(self, x):
        xx = self.x2xx(x)
        k = self.xx2k(xx)
        y = self.k2y(k)
        return y
    
    def adjoint(self, y):
        k = self.y2k(y)
        xx = self.k2xx(k)
        x = self.xx2x(xx)
        return x
    
    def x2xx(self, x):
        xx = self.sn*x
        return xx
    
    def xx2k(self,xx):
        k = self.torch.zeros(self.Kd, dtype=self.torch.complex64).to(self.device)
        k[list(slice(self.Nd[jj]) for jj in range(0, self.ndims))] = xx
        k = self.torch.fft.fftn(k)
        return k
    
    def k2y(self, k):
        torch_y = self.torch_sp.mv(self.torch.flatten(k)[self.mask_G])
        return torch_y
    
    def y2k(self, torch_y):
        k_tmp = self.torch.zeros(numpy.prod(self.Kd), dtype=self.torch.complex64).to(self.device)
        k_tmp[self.mask_G] = self.torch_spH.mv(torch_y)
        k = self.torch.reshape(k_tmp, self.Kd)
        return k
    
    def k2xx(self, k):
        xx = self.torch.fft.ifftn(k, norm="forward")[list(slice(self.Nd[jj]) for jj in range(0, self.ndims))] 
        return xx
    
    def xx2x(self,xx):
        return self.x2xx(xx)