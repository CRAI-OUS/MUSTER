import torch
import torch.nn.functional as F

import numpy as np
import math

from . import field_calculus


class NCC:
    """Normalized Local Cross Correlation.

    Args:
        kernal_size (int): The size of the kernal used for calculating the local mean
    """

    def __init__(self, kernal_size=7, reduce=True, scale_invariant=True):
        self.kernal_size = kernal_size
        self.n = kernal_size**3
        self.padding = kernal_size // 2
        self.reduce = reduce
        self.scale_invariant = scale_invariant

    def loss(self, y_true, y_pred, mask=None):
        """
        Args:
            y_true (torch.Tensor): The true field
            y_pred (torch.Tensor): The predicted field
            mask (torch.Tensor): The mask used for calculating the loss
        """
        f_i = y_true
        g_i = y_pred

        ff = f_i * f_i
        gg = g_i * g_i
        fg = f_i * g_i

        f_bar = torch.nn.functional.avg_pool3d(f_i, self.kernal_size, stride=1, padding=self.padding)
        g_bar = torch.nn.functional.avg_pool3d(g_i, self.kernal_size, stride=1, padding=self.padding)
        sum_ff = torch.nn.functional.avg_pool3d(ff, self.kernal_size, stride=1, padding=self.padding)
        sum_gg = torch.nn.functional.avg_pool3d(gg, self.kernal_size, stride=1, padding=self.padding)
        sum_fg = torch.nn.functional.avg_pool3d(fg, self.kernal_size, stride=1, padding=self.padding)

        cross = sum_fg - f_bar * g_bar
        f_var = sum_ff - f_bar * f_bar
        g_var = sum_gg - g_bar * g_bar

        # Masking
        if mask is not None:
            cross = cross * mask
            f_var = f_var * mask
            g_var = g_var * mask

        if not self.scale_invariant:
            return (1 - torch.mean(cross * cross / (f_var * g_var + 0.0001)))
        else:
            return torch.mean((f_var + 0.0001) - cross * cross / (g_var + 0.0001))


def fourier_filter(img, filter_response, dims=(-3, -2, -1)):
    return torch.fft.ifftn(torch.fft.fftn(img, dim=(-3, -2, -1)) * filter_response, dim=(-3, -2, -1)).real


class LogLinLoss:
    """Assuming I_ix = a_i I_x e find the I_ix that maximizes the likelihood of this model
    
    Args:
        kernal_size (int): The size of the kernal used for calculating the local mean
    """

    def __init__(self,
                 kernal_size=7,
                 kernal_type="gaussian",
                 pix_dim=(1, 1, 1),
                 img_size=(256, 256, 256),
                 log_transform=True):
        self.kernal_type = kernal_type
        if kernal_type == "window":
            # Check that the kernal size is an integer or an float close to an integer
            if not (isinstance(kernal_size, int) or math.isclose(kernal_size, int(kernal_size))):
                raise ValueError("The kernal size must be an integer or an float close to an integer")
            # Check that the kernal size is odd
            if kernal_size % 2 == 0:
                raise ValueError("The kernal size must be odd")
            self.kernal_size = int(kernal_size)
            self.n = kernal_size**3
            self.padding = kernal_size // 2
        elif kernal_type == "gaussian":
            self.omega_f = 1 / (2 * torch.pi * kernal_size)
            self.pix_dim = pix_dim
            self.img_size = img_size
        self.log_transform = log_transform

    def loss(self, images, masks=None, sigma_beta=1.0):
        """
        Args:
            images (torch.Tensor): the images warped to the same space
            mask (torch.Tensor): The mask used for calculating the loss
        """
        #masks = images != 0
        if masks is None:
            masks = torch.ones_like(images)

        masks = torch.prod(masks, dim=0, keepdim=True)
        masks = masks.float()

        eps = 1e-5
        if self.log_transform:
            f_i = torch.log(images + eps) * masks
        else:
            f_i = images * masks

        if self.kernal_type == "window":
            # The number of valid voxels in the kernal
            n_kernal_f = torch.nn.functional.avg_pool3d(
                masks, self.kernal_size, stride=1, padding=self.padding) * self.n
            # Make sure we don't divide by zero
            n_kernal_f = torch.max(n_kernal_f, torch.ones_like(n_kernal_f)) + 1 / sigma_beta**2

            # Calculate the local mean
            E_f = torch.nn.functional.avg_pool3d(
                f_i, self.kernal_size, stride=1, padding=self.padding) * self.n / n_kernal_f

        elif self.kernal_type == "gaussian":
            # Gaussian filter
            freqsx = torch.fft.fftfreq(self.img_size[0], d=self.pix_dim[0], device=images.device)
            freqsy = torch.fft.fftfreq(self.img_size[1], d=self.pix_dim[1], device=images.device)
            freqsz = torch.fft.fftfreq(self.img_size[2], d=self.pix_dim[2], device=images.device)
            freqsx, freqsy, freqsz = torch.meshgrid(freqsx, freqsy, freqsz, indexing="ij")
            filter_response = torch.exp(-1 / 2 * ((freqsx**2 + freqsy**2 + freqsz**2) / self.omega_f**2))

            n_kernal_f = fourier_filter(masks, filter_response)
            n_kernal_f = torch.max(n_kernal_f, torch.ones_like(n_kernal_f)) + 1 / sigma_beta**2

            # Calculate the local mean
            E_f = fourier_filter(f_i, filter_response) / n_kernal_f

        f_i_bar = f_i - E_f
        E_f_i_bar = torch.mean(f_i_bar, dim=0, keepdim=True)
        return torch.mean((f_i_bar - E_f_i_bar)**2 * masks) / torch.mean(masks), E_f_i_bar


class GroupVELLN:
    """Variance Estimating Local Linear compansated Normal distribution loss
    
    Args:
        kernal_size (int): The size of the kernal used for calculating the local mean
    """

    def __init__(self, kernal_size=7, kernal_type="gaussian", pix_dim=(1, 1, 1), img_size=(256, 256, 256)):
        self.kernal_type = kernal_type
        if kernal_type == "window":
            # Check that the kernal size is an integer or an float close to an integer
            if not (isinstance(kernal_size, int) or math.isclose(kernal_size, int(kernal_size))):
                raise ValueError("The kernal size must be an integer or an float close to an integer")
            # Check that the kernal size is odd
            if kernal_size % 2 == 0:
                raise ValueError("The kernal size must be odd")
            self.kernal_size = int(kernal_size)
            self.n = kernal_size**3
            self.padding = kernal_size // 2
        elif kernal_type == "gaussian":
            self.omega_f = 1 / (2 * torch.pi * kernal_size)
            self.pix_dim = pix_dim
            self.img_size = img_size

    def loss(self, images, masks=None):
        """
        Args:
            y_true (torch.Tensor): The true field
            y_pred (torch.Tensor): The predicted field
            mask (torch.Tensor): The mask used for calculating the loss
        """
        #masks = images != 0
        if masks is None:
            masks = torch.ones_like(images)
        masks = torch.prod(masks, dim=0, keepdim=True)
        masks = masks.float()

        f_i = images * masks

        #with torch.no_grad():
        ff = f_i * f_i

        if self.kernal_type == "window":
            # The number of valid voxels in the kernal
            n_kernal_f = torch.nn.functional.avg_pool3d(
                masks, self.kernal_size, stride=1, padding=self.padding) * self.n
            # Make sure we don't divide by zero
            n_kernal_f = torch.max(n_kernal_f, torch.ones_like(n_kernal_f))

            # Calculate the local mean
            E_f = torch.nn.functional.avg_pool3d(
                f_i, self.kernal_size, stride=1, padding=self.padding) * self.n / n_kernal_f
            E_ff = torch.nn.functional.avg_pool3d(
                ff, self.kernal_size, stride=1, padding=self.padding) * self.n / n_kernal_f

        elif self.kernal_type == "gaussian":
            # Gaussian filter
            freqsx = torch.fft.fftfreq(self.img_size[0], d=self.pix_dim[0], device=images.device)
            freqsy = torch.fft.fftfreq(self.img_size[1], d=self.pix_dim[1], device=images.device)
            freqsz = torch.fft.fftfreq(self.img_size[2], d=self.pix_dim[2], device=images.device)
            freqsx, freqsy, freqsz = torch.meshgrid(freqsx, freqsy, freqsz, indexing="ij")
            filter_response = torch.exp(-1 / 2 * ((freqsx**2 + freqsy**2 + freqsz**2) / self.omega_f**2))

            n_kernal_f = fourier_filter(masks, filter_response)
            n_kernal_f = torch.max(n_kernal_f, torch.ones_like(n_kernal_f))

            # Calculate the local mean
            E_f = fourier_filter(f_i, filter_response) / n_kernal_f
            E_ff = fourier_filter(ff, filter_response) / n_kernal_f

        f_var = E_ff - E_f * E_f + 0.0001
        f_i_bar = (f_i - E_f)
        f_i_bar = f_i
        I = torch.mean(f_i_bar / torch.sqrt(f_var), dim=0, keepdim=True)
        return torch.mean((f_i_bar - torch.sqrt(f_var) * I)**2 * masks) / torch.mean(masks), I * torch.mean(
            torch.sqrt(f_var), dim=0, keepdim=True)  #+ torch.mean(E_f, dim=0, keepdim=True)


class VELLN:
    """Variance Estimating Local Linear compansated Normal distribution loss
    
    Args:
        kernal_size (int): The size of the kernal used for calculating the local mean
    """

    def __init__(self, kernal_size=7, kernal_type="gaussian", pix_dim=(1, 1, 1), img_size=(256, 256, 256)):
        self.kernal_type = kernal_type
        if kernal_type == "window":
            # Check that the kernal size is an integer or an float close to an integer
            if not (isinstance(kernal_size, int) or math.isclose(kernal_size, int(kernal_size))):
                raise ValueError("The kernal size must be an integer or an float close to an integer")
            # Check that the kernal size is odd
            if kernal_size % 2 == 0:
                raise ValueError("The kernal size must be odd")
            self.kernal_size = int(kernal_size)
            self.n = kernal_size**3
            self.padding = kernal_size // 2
        elif kernal_type == "gaussian":
            self.omega_f = 1 / (2 * torch.pi * kernal_size)
            self.pix_dim = pix_dim
            self.img_size = img_size

    def loss(self, y_true, y_pred, sigma_true, sigma_pred, mask=None):
        """
        Args:
            y_true (torch.Tensor): The true field
            y_pred (torch.Tensor): The predicted field
            mask (torch.Tensor): The mask used for calculating the loss
        """
        #mask_true = y_true != 0
        mask_true = torch.ones_like(y_true)
        mask_true = mask_true.float()

        #mask_pred = y_pred != 0
        mask_pred = torch.ones_like(y_pred)
        mask_pred = mask_pred.float()

        f_i = y_true
        g_i = y_pred

        #with torch.no_grad():
        ff = f_i * f_i
        gg = g_i * g_i
        fg = f_i * g_i

        if self.kernal_type == "window":
            # The number of valid voxels in the kernal
            n_kernal_f = torch.nn.functional.avg_pool3d(
                mask_true, self.kernal_size, stride=1, padding=self.padding) * self.n
            # Make sure we don't divide by zero
            n_kernal_f = torch.max(n_kernal_f, torch.ones_like(n_kernal_f))

            n_kernal_g = torch.nn.functional.avg_pool3d(
                mask_pred, self.kernal_size, stride=1, padding=self.padding) * self.n
            n_kernal_g = torch.max(n_kernal_g, torch.ones_like(n_kernal_g))

            n_kernal_fg = torch.nn.functional.avg_pool3d(
                mask_true * mask_pred, self.kernal_size, stride=1, padding=self.padding) * self.n
            n_kernal_fg = torch.max(n_kernal_fg, torch.ones_like(n_kernal_fg))

            # Calculate the local mean
            E_f = torch.nn.functional.avg_pool3d(
                f_i, self.kernal_size, stride=1, padding=self.padding) * self.n / n_kernal_f
            E_g = torch.nn.functional.avg_pool3d(
                g_i, self.kernal_size, stride=1, padding=self.padding) * self.n / n_kernal_g
            E_ff = torch.nn.functional.avg_pool3d(
                ff, self.kernal_size, stride=1, padding=self.padding) * self.n / n_kernal_f
            E_gg = torch.nn.functional.avg_pool3d(
                gg, self.kernal_size, stride=1, padding=self.padding) * self.n / n_kernal_g
            E_fg = torch.nn.functional.avg_pool3d(
                fg, self.kernal_size, stride=1, padding=self.padding) * self.n / n_kernal_fg
        elif self.kernal_type == "gaussian":
            # Gaussian filter
            freqsx = torch.fft.fftfreq(self.img_size[0], d=self.pix_dim[0], device=y_true.device)
            freqsy = torch.fft.fftfreq(self.img_size[1], d=self.pix_dim[1], device=y_true.device)
            freqsz = torch.fft.fftfreq(self.img_size[2], d=self.pix_dim[2], device=y_true.device)
            freqsx, freqsy, freqsz = torch.meshgrid(freqsx, freqsy, freqsz, indexing="ij")
            filter_response = torch.exp(-1 / 2 * ((freqsx**2 + freqsy**2 + freqsz**2) / self.omega_f**2))

            n_kernal_f = fourier_filter(mask_true, filter_response)
            n_kernal_f = torch.max(n_kernal_f, torch.ones_like(n_kernal_f))
            n_kernal_g = fourier_filter(mask_pred, filter_response)
            n_kernal_g = torch.max(n_kernal_g, torch.ones_like(n_kernal_g))
            n_kernal_fg = fourier_filter(mask_true * mask_pred, filter_response)
            n_kernal_fg = torch.max(n_kernal_fg, torch.ones_like(n_kernal_fg))

            # Calculate the local mean
            E_f = fourier_filter(f_i, filter_response) / n_kernal_f
            E_g = fourier_filter(g_i, filter_response) / n_kernal_g
            E_ff = fourier_filter(ff, filter_response) / n_kernal_f
            E_gg = fourier_filter(gg, filter_response) / n_kernal_g
            E_fg = fourier_filter(fg, filter_response) / n_kernal_fg

        f_var = E_ff - E_f * E_f + 0.0001
        g_var = E_gg - E_g * E_g + 0.0001
        fg_cov = E_fg - E_f * E_g + 0.0001

        # Assuming sigma_true << 1
        a = fg_cov / f_var
        # b = E_g - a * E_f

        diff_2 = g_var - fg_cov**2 / f_var

        scale_2 = a**2 * torch.exp(2 * sigma_true) + torch.exp(2 * sigma_pred)
        log_scale = 0.5 * torch.log(scale_2)

        return torch.sum((0.5 * diff_2 / scale_2 + log_scale) * mask_true) / torch.sum(mask_true)


# class VELLN:
#     """Variance Estimating Local Linear compansated Normal distribution loss

#     Args:
#         kernal_size (int): The size of the kernal used for calculating the local mean
#     """

#     def __init__(self, kernal_size=7):
#         self.kernal_size = kernal_size
#         self.n = kernal_size**3
#         self.padding = kernal_size // 2

#     def loss(self, y_true, y_pred, sigma_true, sigma_pred, mask=None):
#         """
#         Args:
#             y_true (torch.Tensor): The true field
#             y_pred (torch.Tensor): The predicted field
#             mask (torch.Tensor): The mask used for calculating the loss
#         """
#         #mask_true = y_true != 0
#         mask_true = torch.ones_like(y_true)
#         mask_true = mask_true.float()

#         #mask_pred = y_pred != 0
#         mask_pred = torch.ones_like(y_pred)
#         mask_pred = mask_pred.float()

#         f_i = y_true
#         g_i = y_pred

#         with torch.no_grad():
#             ff = f_i * f_i
#             gg = g_i * g_i

#             # The number of valid voxels in the kernal
#             n_kernal_f = torch.nn.functional.avg_pool3d(
#                 mask_true, self.kernal_size, stride=1, padding=self.padding) * self.n
#             # Make sure we don't divide by zero
#             n_kernal_f = torch.max(n_kernal_f, torch.ones_like(n_kernal_f))

#             n_kernal_g = torch.nn.functional.avg_pool3d(
#                 mask_pred, self.kernal_size, stride=1, padding=self.padding) * self.n
#             n_kernal_g = torch.max(n_kernal_g, torch.ones_like(n_kernal_g))

#             # Calculate the local mean
#             E_f = torch.nn.functional.avg_pool3d(
#                 f_i, self.kernal_size, stride=1, padding=self.padding) * self.n / n_kernal_f
#             E_g = torch.nn.functional.avg_pool3d(
#                 g_i, self.kernal_size, stride=1, padding=self.padding) * self.n / n_kernal_g
#             E_ff = torch.nn.functional.avg_pool3d(
#                 ff, self.kernal_size, stride=1, padding=self.padding) * self.n / n_kernal_f
#             E_gg = torch.nn.functional.avg_pool3d(
#                 gg, self.kernal_size, stride=1, padding=self.padding) * self.n / n_kernal_g

#             f_var = E_ff - E_f * E_f + 0.0001
#             f_sigma = torch.sqrt(f_var)

#             g_var = E_gg - E_g * E_g + 0.0001
#             g_sigma = torch.sqrt(g_var)

#             sigma_f_g = f_sigma / (g_sigma + 0.000001)

#         diff = (f_i - E_f) - sigma_f_g * (g_i - E_g)
#         scale = torch.exp(sigma_true) + sigma_f_g * torch.exp(sigma_pred)
#         log_scale = torch.log(scale)

#         # return torch.sum(
#         #     (0.5 * (((f_i - E_f) - sigma_f_g * (g_i - E_g)) /
#         #             (sigma_true + sigma_f_g * sigma_pred))**2 + torch.log(sigma_true + sigma_f_g * sigma_pred)) *
#         #     mask) / torch.sum(mask)
#         return torch.sum((0.5 * (diff / scale)**2 + log_scale) * mask_true) / torch.sum(mask_true)


class GaussNCC:

    def __init__(self, filter_omega, img_size, pix_dim, reduce=True, scale_invariant=True):
        self.filter_omega = filter_omega  #/pix_dim[0]
        self.img_size = img_size
        self.pix_dim = pix_dim
        self.omega_f = 1 / (2 * torch.pi * filter_omega)
        self.reduce = reduce
        self.scale_invariant = scale_invariant

    def loss(self, y_true, y_pred, mask=None):

        f_i = y_true
        g_i = y_pred

        ff = f_i * f_i
        gg = g_i * g_i
        fg = f_i * g_i

        # Gaussian filter
        freqsx = torch.fft.fftfreq(self.img_size[0], d=self.pix_dim[0], device=y_true.device)
        freqsy = torch.fft.fftfreq(self.img_size[1], d=self.pix_dim[1], device=y_true.device)
        freqsz = torch.fft.fftfreq(self.img_size[2], d=self.pix_dim[2], device=y_true.device)
        freqsx, freqsy, freqsz = torch.meshgrid(freqsx, freqsy, freqsz, indexing="ij")
        filter_response = torch.exp(-1 / 2 * ((freqsx**2 + freqsy**2 + freqsz**2) / self.omega_f**2))

        f_bar = torch.fft.ifftn(torch.fft.fftn(f_i, dim=(-3, -2, -1)) * filter_response, dim=(-3, -2, -1)).real
        g_bar = torch.fft.ifftn(torch.fft.fftn(g_i, dim=(-3, -2, -1)) * filter_response, dim=(-3, -2, -1)).real
        sum_ff = torch.fft.ifftn(torch.fft.fftn(ff, dim=(-3, -2, -1)) * filter_response, dim=(-3, -2, -1)).real
        sum_gg = torch.fft.ifftn(torch.fft.fftn(gg, dim=(-3, -2, -1)) * filter_response, dim=(-3, -2, -1)).real
        sum_fg = torch.fft.ifftn(torch.fft.fftn(fg, dim=(-3, -2, -1)) * filter_response, dim=(-3, -2, -1)).real

        cross = sum_fg - f_bar * g_bar
        f_var = sum_ff - f_bar * f_bar
        g_var = sum_gg - g_bar * g_bar

        # Masking
        if mask is not None:
            cross = cross * mask
            f_var = f_var * mask
            g_var = g_var * mask

        if not self.scale_invariant:
            return (1 - torch.mean(cross * cross / (f_var * g_var + 0.0001)))
        else:
            f_var = f_var + 0.001
            g_var = g_var + 0.001
            return torch.mean(f_var - cross * cross / (g_var))


class Fourier:
    """Spatial jacobian loss using the fourier space.
    
    Args:
        filter_omega (float): The cutoff frequency of the filter
        img_size (tuple): The size of the image
        pix_dim (tuple): The pixel dimensions of the image
    """

    def __init__(self, filter_omega, img_size, pix_dim):
        self.filter_omega = filter_omega  #/pix_dim[0]
        self.img_size = img_size
        self.pix_dim = pix_dim

    def loss(self, y_true, y_pred, mask=None):
        filtered_true = self.filter(y_true)
        filtered_pred = self.filter(y_pred)

        if mask is not None:
            filtered_true = filtered_true * mask
            filtered_pred = filtered_pred * mask

        dims = [-4, -3, -2, -1]
        diff_true = filtered_true  #- torch.mean(filtered_true, dims, keepdim=True)
        diff_pred = filtered_pred  #- torch.mean(filtered_pred, dims, keepdim=True)
        return 1 - torch.mean(
            torch.sum(diff_true * diff_pred, dims) /
            torch.sqrt(torch.sum(diff_true * diff_true, dims) * torch.sum(diff_pred * diff_pred, dims)))

    def filter(self, y):
        y_fft = torch.fft.fftn(y, dim=(-3, -2, -1))

        # Compute the derivative by multiplying with the fourier space coordinates
        freqsx = torch.fft.fftfreq(self.img_size[0], d=self.pix_dim[0], device=y.device)
        freqsy = torch.fft.fftfreq(self.img_size[1], d=self.pix_dim[1], device=y.device)
        freqsz = torch.fft.fftfreq(self.img_size[2], d=self.pix_dim[2], device=y.device)
        freqsx, freqsy, freqsz = torch.meshgrid(freqsx, freqsy, freqsz, indexing="ij")

        filter = torch.exp(-1 / 2 * (freqsx**2 + freqsy**2 + freqsz**2) / self.filter_omega**2)

        freqs = torch.stack([freqsx, freqsy, freqsz], dim=0).to(y.device)

        filter = filter * freqs * 1j

        y_fft = y_fft * filter[None, ...]
        y_fft = torch.fft.ifftn(y_fft, dim=(-3, -2, -1)).real
        return y_fft


class NCCS:
    """ NCC with Sobel filter.
    See: https://doi.org/10.3390/app12062828
    
    Args:
        kernal_size (int): The size of the kernal used for calculating the local mean
    """

    def __init__(self, kernal_size=13):
        self.kernal_size = kernal_size
        self.n = kernal_size**3
        self.padding = kernal_size // 2

    def loss(self, y_true, y_pred, mask=None):

        vox_img = torch.prod(torch.tensor(y_true.shape[2:]))

        y_true = field_calculus.jacobian(y_true, filter=True, midpoint=True)[:, :, 0, ...]
        y_pred = field_calculus.jacobian(y_pred, filter=True, midpoint=True)[:, :, 0, ...]

        y_true_means = torch.nn.functional.avg_pool3d(y_true, self.kernal_size, stride=1, padding=self.padding)
        y_pred_means = torch.nn.functional.avg_pool3d(y_pred, self.kernal_size, stride=1, padding=self.padding)
        diff_true = y_true - y_true_means
        diff_pred = y_pred - y_pred_means

        diff_true_squared = torch.nn.functional.avg_pool3d(
            diff_true * diff_true, self.kernal_size, stride=1, padding=self.padding)
        diff_pred_squared = torch.nn.functional.avg_pool3d(
            diff_pred * diff_pred, self.kernal_size, stride=1, padding=self.padding)

        diff_true_pred = torch.nn.functional.avg_pool3d(
            diff_true * diff_pred, self.kernal_size, stride=1, padding=self.padding)

        if mask is not None:
            diff_true_squared = diff_true_squared * mask
            diff_pred_squared = diff_pred_squared * mask
            diff_true_pred = diff_true_pred * mask

        return 1 - torch.sum(
            diff_true_pred / torch.sqrt(diff_true_squared * diff_pred_squared + 0.0001)) / torch.sum(mask)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred, mask=None):

        if mask is not None:
            return torch.mean((y_true - y_pred)**2 * mask) / torch.mean(mask)
        else:
            return torch.mean((y_true - y_pred)**2)


class MutualInformation(torch.nn.Module):

    def __init__(self, sigma=0.1, num_bins=256, normalize=True, reduce=True):
        super(MutualInformation, self).__init__()

        self.sigma = sigma
        self.num_bins = num_bins
        self.normalize = normalize
        self.epsilon = 1e-10

        self.bins = torch.nn.Parameter(torch.linspace(0, 1, num_bins).float(), requires_grad=False)
        self.reduce = reduce

    def marginalPdf(self, values):
        with torch.no_grad():
            # Get the max value for each image
            v_max = torch.max(values)

        print(f"Vmax: {v_max}")

        residuals = values - self.bins.unsqueeze(0).unsqueeze(0) / v_max
        kernel_values = torch.exp(-0.5 * (residuals / self.sigma).pow(2))

        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.epsilon
        pdf = pdf / normalization

        import matplotlib.pyplot as plt
        plt.plot(pdf[0, :].cpu().numpy())
        plt.show()

        return pdf, kernel_values

    def jointPdf(self, kernel_values1, kernel_values2):

        joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2)
        normalization = torch.sum(joint_kernel_values, dim=(1, 2)).view(-1, 1, 1) + self.epsilon
        pdf = joint_kernel_values / normalization

        return pdf

    def getMutualInformation(self, input1, input2):
        '''
			input1: B, C, H, W, D
			input2: B, C, H, W, D

			return: scalar
		'''

        B, C, H, W, D = input1.shape
        assert ((input1.shape == input2.shape))

        input1 = input1.permute(0, 2, 3, 4, 1).contiguous()
        input2 = input2.permute(0, 2, 3, 4, 1).contiguous()

        x1 = input1.view(B, H * W * D, C)
        x2 = input2.view(B, H * W * D, C)

        pdf_x1, kernel_values1 = self.marginalPdf(x1)
        pdf_x2, kernel_values2 = self.marginalPdf(x2)
        pdf_x1x2 = self.jointPdf(kernel_values1, kernel_values2)

        H_x1 = -torch.sum(pdf_x1 * torch.log2(pdf_x1 + self.epsilon), dim=1)
        H_x2 = -torch.sum(pdf_x2 * torch.log2(pdf_x2 + self.epsilon), dim=1)
        H_x1x2 = -torch.sum(pdf_x1x2 * torch.log2(pdf_x1x2 + self.epsilon), dim=(1, 2))

        mutual_information = H_x1 + H_x2 - H_x1x2

        if self.normalize:
            mutual_information = 2 * mutual_information / (H_x1 + H_x2)

        print(mutual_information)

        return mutual_information

    def loss(self, y_true, y_pred):
        if self.reduce:
            return torch.mean(self.getMutualInformation(y_true, y_pred))
        else:
            return self.getMutualInformation(y_true, y_pred)
