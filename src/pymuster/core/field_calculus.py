""" Containing useful functions for field calculus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def jacobian(x, pix_dim=(1.0, 1.0, 1.0), filter=False, midpoint=False):
    """
    Calculates the Jacobian of a 3D field
    Args:
        x: tensor of shape (batch_size, channels, x, y, z)
    Returns:
        J: jacobian of shape (batch_size, 3, channels, x, y, z)
    """
    m = x.shape[1]
    # Assert that filter only is used with midpoint
    assert not filter or midpoint, "Filter only works with midpoint"

    if filter:
        h = torch.tensor([[[1, 2, 1]]], dtype=torch.float32) / 4
    else:
        h = torch.tensor([[[0, 1, 0]]], dtype=torch.float32)

    if midpoint:
        h_m = torch.tensor([[[1, 0, -1]]], dtype=torch.float32) / 2
    else:
        h_m = torch.tensor([[[0, 1, -1]]], dtype=torch.float32)

    s3_x = torch.einsum('ijk, jki, kij -> kij', h_m / pix_dim[0], h, h)
    s3_y = torch.einsum('ijk, jki, kij -> kij', h, h_m / pix_dim[1], h)
    s3_z = torch.einsum('ijk, jki, kij -> kij', h, h, h_m / pix_dim[2])

    # s3_x = torch.einsum('ijk, jki, kij -> kij', h_m, h, h)
    # s3_y = torch.einsum('ijk, jki, kij -> kij', h, h_m, h)
    # s3_z = torch.einsum('ijk, jki, kij -> kij', h, h, h_m)

    s_filter = torch.cat((s3_x[None, ...], s3_y[None, ...], s3_z[None, ...]), dim=0)

    s_filter = s_filter.repeat(m, 1, 1, 1)[:, None, ...].to(x.device)

    # Use convolutions to calculate the jacobian
    J = F.conv3d(x, s_filter, padding=1, groups=m)
    J = J.reshape(J.shape[0], 3, m, J.shape[2], J.shape[3], J.shape[4])

    return J

def curl(J):
    """
    Returns the curl of a 3D field
    Args:
        J: Jacobian of shape (batch_size, 3, 3, x, y, z)
    """
    C = torch.zeros(J.shape[0], 3, J.shape[3], J.shape[4], J.shape[5]).to(J.device)
    C[:, 0, :, :, :] = J[:, 2, 1, :, :, :] - J[:, 1, 2, :, :, :]
    C[:, 1, :, :, :] = J[:, 0, 2, :, :, :] - J[:, 2, 0, :, :, :]
    C[:, 2, :, :, :] = J[:, 1, 0, :, :, :] - J[:, 0, 1, :, :, :]
    return C


def divergence(J):
    """
    Returns the divergence of a 3D field
    Args:
        J: Jacobian of shape (batch_size, 3, 3, x, y, z)
    """
    return J[:, 0, 0, :, :, :] + J[:, 1, 1, :, :, :] + J[:, 2, 2, :, :, :]


def laplacian(J, pix_dim=(1.0, 1.0, 1.0)):
    """
    Return the second order jacobian of a field
    """
    channels = J.shape[2]
    L = jacobian(J.reshape(J.shape[0], 3 * channels, J.shape[3], J.shape[4], J.shape[5]), pix_dim=pix_dim)
    L = L.reshape(J.shape[0], channels, 3, 3, J.shape[3], J.shape[4], J.shape[5])
    return L


def determinant(J):
    """
    Computes the determinant of a 3x3 matrix
    """
    J = J.permute(0, 3, 4, 5, 1, 2)
    return torch.det(J)


def integrate_forward_flow(v, sp_tr, n_iter=5, dt=1):
    """
    implements step (4): Calculation of forward flows
    @return:
    """
    image_size = v.shape[-3:]
    T = v.shape[0]
    # create flow
    Phi0 = torch.zeros(v.shape, device=v.device)

    # Phi0_0 is the identity mapping
    # create sampling grid
    vectors = [torch.arange(0, s) for s in image_size]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)
    grid = torch.unsqueeze(grid, 0)
    print(f"Grid shape: {grid.shape}")

    # Phi0_0 is the identity mapping
    Phi0[0] = grid[0]

    for t in range(0, T - 1):
        alpha = forward_alpha(v[[t]], sp_tr, n_iter, dt)
        Phi0[t + 1] = sp_tr(Phi0[[t]], -alpha)

    return Phi0


def forward_alpha(v_t, sp_tr, n_iter, dt=1):
    """
    helper function for step (4): Calculation of forward flows
    @param v_t: the velocity field
    @param sp_tr: the spatial transformer
    @return:
    """
    alpha = torch.zeros(v_t.shape, device=v_t.device)
    for i in range(n_iter):
        alpha = dt * sp_tr(v_t, -0.5 * alpha)
    return alpha
