import torch
from torch import nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SSIMLoss2D(nn.Module):
    """
    2D SSIM loss module.

    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        #assert isinstance(self.w, torch.Tensor)

        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / (D + 1e-8)

        return 1 - S.mean()


class SSIMLoss2D_MC(nn.Module):
    """
    2D multichannel SSIM loss module.

    """

    def __init__(self, win_size: int=7, k1: float=0.01, k2: float=0.03, in_chan: int=1):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
            in_chan: number of input channels
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.in_chan = in_chan
        self.register_buffer("w", torch.ones(in_chan, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        #assert isinstance(self.w, torch.Tensor)


        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w.cuda(), groups=self.in_chan)  # per-channel convolution
        uy = F.conv2d(Y, self.w.cuda(), groups=self.in_chan)  #
        uxx = F.conv2d(X * X, self.w.cuda(), groups=self.in_chan)
        uyy = F.conv2d(Y * Y, self.w.cuda(), groups=self.in_chan)
        uxy = F.conv2d(X * Y, self.w.cuda(), groups=self.in_chan)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / (D + 1e-8)

        return 1 - S.mean()


class SSIMLoss3D(nn.Module):
    """
    3D SSIM loss module

    Square window with uniform weights (non-Gaussian) following the implementation
    from Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). 
    Image quality assessment: From error visibility to structural similarity.
    IEEE Transactions on Image Processing, 13, 600-612.
    https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
    :DOI:`10.1109/TIP.2003.819861` with slight modification for efficiency

    by Vick Lau 2021
    """

    def __init__(self, win_size: int=7, k1: float=0.01, k2: float=0.03):
        """Initialise 3D SSIM loss

        Args
        -------
            win_size (int): Window size
            k1 (float): k1 parameter
            k2 (float): k2 parameter
        Returns
        -------
            torch.Tensor: 3D SSIM Loss 
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size, win_size) / win_size**3)
        NP = win_size ** 3
        self.cov_norm = NP / (NP - 1) # sample covariance instead of population

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        #assert isinstance(self.w, torch.Tensor)
    
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2

        # compute variances and covariances
        ux = F.conv3d(X, self.w)
        uy = F.conv3d(Y, self.w)
        uxx = F.conv3d(X * X, self.w)
        uyy = F.conv3d(Y * Y, self.w)
        uxy = F.conv3d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux*ux)
        vy = self.cov_norm * (uyy - uy*uy)
        vxy = self.cov_norm * (uxy - ux*uy)
        A1, A2, B1, B2 = (
            2*ux*uy + C1,
            2*vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1*B2
        S = (A1*A2) / (D + 1e-8) # eps for floating point stability

        return 1 - S.mean() # compute 1 - mean of SSIM
