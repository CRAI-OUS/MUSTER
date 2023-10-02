import torch
import torch.nn.functional as F

import numpy as np
import math

from . import field_calculus


class Fourier:

    def __init__(self, filter_omega, image_size, pix_dim):
        self.filter_omega = filter_omega  #/pix_dim[0]
        self.image_size = image_size
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
        freqsx = torch.fft.fftfreq(self.image_size[0], d=self.pix_dim[0], device=y.device)
        freqsy = torch.fft.fftfreq(self.image_size[1], d=self.pix_dim[1], device=y.device)
        freqsz = torch.fft.fftfreq(self.image_size[2], d=self.pix_dim[2], device=y.device)
        freqsx, freqsy, freqsz = torch.meshgrid(freqsx, freqsy, freqsz)

        filter = torch.exp(-1 / 2 * (freqsx**2 + freqsy**2 + freqsz**2) / self.filter_omega**2)

        freqs = torch.stack([freqsx, freqsy, freqsz], dim=0).to(y.device)

        filter = filter * freqs * 1j

        y_fft = y_fft * filter[None, ...]
        y_fft = torch.fft.ifftn(y_fft, dim=(-3, -2, -1)).real
        return y_fft


class GlobalPearsonCorrelation:

    def __init__(self):
        pass

    def loss(self, y_true, y_pred, mask=None):
        if mask is not None:
            y_true = y_true * mask
            y_pred = y_pred * mask
        diff_true = y_true - torch.mean(y_true, [-3, -2, -1], keepdim=True)
        diff_pred = y_pred - torch.mean(y_pred, [-3, -2, -1], keepdim=True)
        return 1 - torch.mean(
            torch.sum(diff_true * diff_pred, [-4, -3, -2, -1]) / torch.sqrt(
                torch.sum(diff_true * diff_true, [-4, -3, -2, -1]) *
                torch.sum(diff_pred * diff_pred, [-4, -3, -2, -1])))


class NCCS:

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


class SemiGlobalPearsonCorrelation:

    def __init__(self, kernal_size=13):
        self.kernal_size = kernal_size
        self.n = kernal_size**3
        self.padding = kernal_size // 2

    def loss(self, y_true, y_pred):

        vox_img = torch.prod(torch.tensor(y_true.shape[2:]))

        #y_true_means = torch.nn.functional.avg_pool3d(y_true, self.kernal_size, stride=1, padding=self.padding)
        #y_pred_means = torch.nn.functional.avg_pool3d(y_pred, self.kernal_size, stride=1, padding=self.padding)
        #diff_true = y_true - y_true_means
        #diff_pred = y_pred - y_pred_means
        diff_true = y_true - torch.mean(y_true, list(range(1, len(y_true.shape))), keepdim=True)
        diff_pred = y_pred - torch.mean(y_pred, list(range(1, len(y_pred.shape))), keepdim=True)
        #diff_true_squared = torch.nn.functional.avg_pool3d(diff_true * diff_true, self.kernal_size, stride=1, padding=self.padding)*self.n
        #diff_pred_squared = torch.nn.functional.avg_pool3d(diff_pred * diff_pred, self.kernal_size, stride=1, padding=self.padding)*self.n
        diff_true_squared = torch.mean(diff_true * diff_true, list(range(1, len(y_true.shape))), keepdim=True) * vox_img
        diff_pred_squared = torch.mean(diff_pred * diff_pred, list(range(1, len(y_pred.shape))), keepdim=True) * vox_img

        diff_true_pred = torch.nn.functional.avg_pool3d(
            diff_true * diff_pred, self.kernal_size, stride=1, padding=self.padding)

        return 1 - torch.mean(diff_true_pred * vox_img / torch.sqrt(diff_true_squared * diff_pred_squared + 0.01))


class PearsonCorrelation:

    def __init__(self, kernal_size=7):
        self.kernal_size = kernal_size
        self.n = kernal_size**3
        self.padding = kernal_size // 2

    def loss(self, y_true, y_pred, mask):
        vox_img = torch.prod(torch.tensor(y_true.shape[2:]))

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
            diff_true = diff_true * mask
            diff_pred = diff_pred * mask
            diff_true_squared = diff_true_squared * mask
            diff_pred_squared = diff_pred_squared * mask
            diff_true_pred = diff_true_pred * mask

        # return 1 - torch.sum(
        #     diff_true_pred / torch.sqrt(diff_true_squared * diff_pred_squared + 0.0001)) / torch.sum(mask)
        return 1 / 2.0 * torch.sum(
            (diff_true / torch.sqrt(diff_true_squared + 0.01) - diff_pred / torch.sqrt(diff_pred_squared + 0.01))**
            2) / torch.sum(mask)


# class PearsonCorrelation:

#     def __init__(self, kernal_size=7):
#         self.kernal_size = kernal_size
#         self.n = kernal_size**3
#         self.padding = kernal_size // 2

#     def loss(self, y_true, y_pred, mask):
#         vox_img = torch.prod(torch.tensor(y_true.shape[2:]))

#         y_true = y_true / y_true.std(dim=[-3, -2, -1], keepdim=True)
#         y_pred = y_pred / y_pred.std(dim=[-3, -2, -1], keepdim=True)

#         y_true_means = torch.nn.functional.avg_pool3d(y_true, self.kernal_size, stride=1, padding=self.padding)
#         y_pred_means = torch.nn.functional.avg_pool3d(y_pred, self.kernal_size, stride=1, padding=self.padding)
#         diff_true = y_true - y_true_means
#         diff_pred = y_pred - y_pred_means

#         diff_true_squared = torch.nn.functional.avg_pool3d(
#             diff_true * diff_true, self.kernal_size, stride=1, padding=self.padding)
#         diff_pred_squared = torch.nn.functional.avg_pool3d(
#             diff_pred * diff_pred, self.kernal_size, stride=1, padding=self.padding)

#         # diff_true_pred = torch.nn.functional.avg_pool3d(
#         #     diff_true * diff_pred, self.kernal_size, stride=1, padding=self.padding)

#         if mask is not None:
#             diff_true_squared = diff_true_squared * mask
#             diff_pred_squared = diff_pred_squared * mask
#             # diff_true_pred = diff_true_pred * mask

#         with torch.no_grad():
#             cross_std = torch.sqrt(torch.sqrt(diff_true_squared * diff_pred_squared + 0.0001))
#             tot_cross_std = torch.sum(cross_std, [-4, -3, -2, -1])

#         # return torch.mean(1 / tot_cross_std * torch.sum(
#         #     cross_std * (diff_true / torch.sqrt(diff_true_squared + 0.00001) -
#         #                  diff_pred / torch.sqrt(diff_pred_squared + 0.00001))**2, [-4, -3, -2, -1]))

#         # return torch.mean(
#         #     torch.sum(torch.sqrt(cross_var) - torch.sqrt(cross_var) * diff_true_pred / cross_var, dim=[-4, -3, -2, -1])
#         #     / torch.sum(mask, dim=[-4, -3, -2, -1]))

#         # Equvalent to the NCC see: B. Panm (2011) Recent Progress in Digital Image Correlation
#         return 1 / 2.0 * torch.mean(
#             (diff_true / torch.sqrt(diff_true_squared + 0.0001) - diff_pred / torch.sqrt(diff_pred_squared + 0.0001))**
#             2)


class FourierPearsonCorrelation:

    def __init__(self, filter_omega, image_size, pix_dim):
        self.filter_omega = filter_omega  #/pix_dim[0]
        self.image_size = image_size
        self.pix_dim = pix_dim

    def loss(self, y_true, y_pred, mask):

        mask = y_true != 0
        mask = mask.float()
        #mask = torch.ones_like(y_true)

        vox_img = torch.prod(torch.tensor(y_true.shape[2:]))

        y_true_f = torch.fft.fftn(y_true, dim=(-3, -2, -1))
        y_pred_f = torch.fft.fftn(y_pred, dim=(-3, -2, -1))

        freqsx = torch.fft.fftfreq(self.image_size[0], d=self.pix_dim[0], device=y_true.device)
        freqsy = torch.fft.fftfreq(self.image_size[1], d=self.pix_dim[1], device=y_true.device)
        freqsz = torch.fft.fftfreq(self.image_size[2], d=self.pix_dim[2], device=y_true.device)
        freqsx, freqsy, freqsz = torch.meshgrid(freqsx, freqsy, freqsz)

        filter = torch.exp(-1 / 2 * (freqsx**2 + freqsy**2 + freqsz**2) / self.filter_omega**2)

        y_true_means_f = filter * y_true_f
        y_pred_means_f = filter * y_pred_f

        diff_true = y_true - torch.fft.ifftn(y_true_means_f, dim=(-3, -2, -1)).real
        diff_pred = y_pred - torch.fft.ifftn(y_pred_means_f, dim=(-3, -2, -1)).real

        diff_true_squared = torch.fft.ifftn(
            filter * torch.fft.fftn(diff_true * diff_true, dim=(-3, -2, -1)), dim=(-3, -2, -1))
        diff_pred_squared = torch.fft.ifftn(
            filter * torch.fft.fftn(diff_pred * diff_pred, dim=(-3, -2, -1)), dim=(-3, -2, -1))
        diff_true_pred = torch.fft.ifftn(
            filter * torch.fft.fftn(diff_true * diff_pred, dim=(-3, -2, -1)), dim=(-3, -2, -1))

        if mask is not None:
            diff_true_squared = diff_true_squared * mask
            diff_pred_squared = diff_pred_squared * mask
            diff_true_pred = diff_true_pred * mask

        # Convert to real
        diff_true_squared = diff_true_squared.real
        diff_pred_squared = diff_pred_squared.real
        diff_true_pred = diff_true_pred.real

        return 1 - torch.sum(
            diff_true_pred / torch.sqrt(diff_true_squared * diff_pred_squared + 0.001)) / torch.sum(mask)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred)**2)
