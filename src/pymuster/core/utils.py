import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from scipy.interpolate import interpn
from scipy.spatial.transform import Rotation as R


def grid(grid_size, grid_interval):
    """
    Constructs a 3d image of a grid with a given size
    For visualization purposes
    Args:
        grid_size: (x, y, z)
        grid_interval: The interval between the grid lines
    """
    image = np.zeros(grid_size)
    # Draw vertical lines
    image[::grid_interval, :, :] = 1
    image[:, ::grid_interval, :] = 1
    image[:, :, ::grid_interval] = 1
    return image


def render_brain(img, theta, chi, N, zoom, gpu=False, alpha_mask=None):
    """
    Renderer for 3d brain output_image
    Args:
        img: 3d brain image [x, y, z]
        theta: viewing angle rotate
        chi: viewing angle up and down
        N: camera grid resolution in x axis
        zoom: zoom factor
    """
    with torch.no_grad():
        # Scale the img to [0, 1]
        # img = (img - img.min()) / (img.max() - img.min())
        # Load the Original img
        if type(img) != torch.Tensor:
            img = torch.from_numpy(img)
        if gpu:
            img = img.cuda()

        img = np.squeeze(img)

        # Construct the Corresponding img Grid Coordinates
        Nx, Ny, Nz = img.shape

        x = torch.linspace(-Nx, Nx, N, device=img.device) / Nx
        y = torch.linspace(-Ny, Ny, N, device=img.device) / Ny
        z = torch.linspace(-Nz, Nz, N, device=img.device) / Nz

        # Apply zoom factor
        x = x * zoom
        y = y * zoom
        z = z * zoom

        qx, qy, qz = torch.meshgrid(x, y, z)  # query points

        # apply rotation
        q = torch.stack([qx.ravel(), qy.ravel(), qz.ravel()], dim=1)

        r = R.from_euler('z', 90, degrees=True)
        r_1 = R.from_euler('x', -theta, degrees=True)
        r_2 = R.from_euler('z', -chi, degrees=True)

        r = r_1 * r * r_2
        r = torch.from_numpy(r.as_matrix()).to(dtype=torch.float32, device=img.device)
        q = torch.matmul(q, r.t())

        # Interpolate onto Camera Grid
        camera_grid = torch.nn.functional.grid_sample(
            img[None, None, :, :, :], q.view(1, N, N, N, 3), mode='bilinear', padding_mode='zeros',
            align_corners=False).squeeze()
        if alpha_mask is not None:
            alpha_mask = torch.from_numpy(alpha_mask).to(dtype=torch.float32, device=img.device)
            alpha_mask = torch.nn.functional.grid_sample(
                alpha_mask[None, None, :, :, :],
                q.view(1, N, N, N, 3),
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False).squeeze()
        # Do Volume Rendering
        output_image = torch.zeros((camera_grid.shape[1], camera_grid.shape[2]), device=img.device)

        for dataslice_idx in range(camera_grid.shape[0]):
            intensity = camera_grid[dataslice_idx, :, :]
            alpha = (camera_grid[dataslice_idx, :, :] * 1.2)**2
            alpha[alpha > 1] = 1
            if alpha_mask is not None:
                alpha = alpha_mask[dataslice_idx, :, :] * alpha
            output_image[:, :] = alpha * intensity + (1 - alpha) * output_image[:, :]  #- 0.005*(output_image[:,:])

        # Scale image to 0-1
        #output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())

    return output_image.cpu().numpy()


def view_brain(img, view_3d=False, gpu=False, vmax=None):
    """
    Visualize brain output_image
    Args:
        img: 3d brain image tensor or numpy array [x, y, z]
        view_3d: if True, show 3d brain image
    """

    if type(img) == torch.Tensor:
        img = img.cpu().numpy()

    img = np.squeeze(img)

    vector_view = False
    if img.shape[0] == 3 and img.ndim == 4:
        # Display as a vector field
        view_2 = plot_vector_field(img, 0, color_wheel=False, vmax=vmax)
        view_1 = plot_vector_field(img, 1, color_wheel=False, vmax=vmax)
        view_3 = plot_vector_field(img, 2, color_wheel=False, vmax=vmax)
        vector_view = True
    else:
        view_2 = img[img.shape[0] // 2, :, :]
        view_1 = img[:, img.shape[1] // 2, :]
        view_3 = img[:, :, img.shape[2] // 2]

    view_1 = np.rot90(view_1, 1)
    view_2 = np.rot90(view_2, 1)
    view_3 = np.rot90(view_3, 1)

    # Pad the output_image such that view_1 and view_2 have the same height
    if view_1.shape[0] < view_2.shape[0]:
        pad = view_2.shape[0] - view_1.shape[0]
        view_1 = np.pad(view_1, ((pad, 0), (0, 0)), mode='constant')
    elif view_1.shape[0] > view_2.shape[0]:
        pad = view_1.shape[0] - view_2.shape[0]
        view_2 = np.pad(view_2, ((pad, 0), (0, 0)), mode='constant')

    # Pad the output_image such that view_1 and view_3 have the same width
    if view_1.shape[1] < view_3.shape[1]:
        pad = view_3.shape[1] - view_1.shape[1]
        view_1 = np.pad(view_1, ((0, 0), (pad, 0)), mode='constant')
    elif view_1.shape[1] > view_3.shape[1]:
        pad = view_1.shape[1] - view_3.shape[1]
        view_3 = np.pad(view_3, ((0, 0), (pad, 0)), mode='constant')

    if view_3d:
        view_4 = render_brain(img, 45, 45, view_2.shape[1], 1.0, gpu)
        if view_3.shape[0] < view_4.shape[0]:
            pad = view_4.shape[0] - view_3.shape[0]
            view_3 = np.pad(view_3, ((pad, 0), (0, 0)), mode='constant')
        elif view_3.shape[0] > view_4.shape[0]:
            pad = view_3.shape[0] - view_4.shape[0]
            view_4 = np.pad(view_4, ((pad, 0), (0, 0)), mode='constant')
    else:
        # Create an empty view with the same width as view_2 and the same height as view_3
        if vector_view:
            view_4 = np.zeros((view_2.shape[1], view_3.shape[0], 3))
            view_4, alpha = get_color_wheel(view_4.shape[1] // 2, view_4.shape[0] // 2, view_4.shape[0] // 5, view_4)
            view_4 = np.rot90(view_4, 1)
            print(view_4.shape)
        else:
            view_4 = np.zeros((view_3.shape[0], view_2.shape[1]))

    # Place the output_image in a 2x2 grid
    # view_1, view_2
    # view_3, view_4

    # Concatenate the output_image
    view_1 = np.concatenate((view_1, view_2), axis=1)
    view_3 = np.concatenate((view_3, view_4), axis=1)
    output_image = np.concatenate((view_1, view_3), axis=0)

    return output_image


def view_stacked_brains(imgs, n, view_3d=False, gpu=False):
    """
    Place n brain views in a row
    Args:
        imgs: tensor or numpy array of shape (n, 1, x, y, z)
        n: number of brain views to place in a row, is set to batch size if n > batch size
        view_3d: if True, render the 3d view. Might be slow
    """
    n = min(n, imgs.shape[0])
    output_image = view_brain(imgs[0, 0, :, :, :], view_3d, gpu)
    for i in range(1, n):
        output_image = np.concatenate((output_image, view_brain(imgs[i, 0, :, :, :], view_3d, gpu)), axis=1)
    return output_image


class RunningMeanVar():
    """
    This class is used to calculate the mean and variance of the n last values
    Example:
        >>> rmv = RunningMeanVar(n=10)
        >>> for i in range(100):
        >>>     rmv.add(i)
        >>>     print(rmv.mean())
        
    Args:
        n: The number of values to use for calculating the mean and variance
    """

    def __init__(self, n=10):
        self.n = n
        self.values = np.zeros(n)
        self.last_index = 0

    def add(self, value):
        """ Add a value to the running mean and variance
        Args:
            value: The value to add
        
        """
        self.values[self.last_index] = value
        self.last_index = (self.last_index + 1) % self.n

    def mean(self):
        """
        Returns: The mean of the last n values
        """
        return np.mean(self.values)

    def var(self):
        """
        Returns: The variance of the last n values
        """
        return np.var(self.values)


def plot_vector_field(
    deformation,
    axis,
    slice_in=None,
    vmax=None,
    color_wheel=True,
):
    """
    Visualize a vector field
    Args:
        deformation: 3d tensor of shape [x, y, z, 3]
        slice_ind: slice to visualize
        axis: axis to visualize
    """

    if slice_in is None:
        slice_ind = deformation.shape[1 + axis] // 2

    #deformation = deformation[[1, 2, 0], :, :, :]

    if axis == 0:
        deformation = deformation[[1, 2], slice_ind, :, :]
    elif axis == 1:
        deformation = deformation[[0, 2], :, slice_ind, :]
    elif axis == 2:
        deformation = deformation[[0, 1], :, :, slice_ind]

    # Calculate the magnitude of the vector field
    magnitude = np.sqrt(np.sum(deformation**2, axis=0))
    # Map magnitude to value
    if vmax is None:
        vmax = magnitude.max()
    value = magnitude / vmax

    # Calculate the angle of the vector field
    angle = np.arctan2(deformation[1], deformation[0])
    # Map angle to hue
    hue = (angle + np.pi) / (2 * np.pi)

    # Set the saturation to 1
    saturation = np.ones_like(hue) * 0.75

    image = np.stack((hue, saturation, value), axis=-1)

    # Add a color wheel

    image = hsv_to_rgb(image)

    if color_wheel:
        scale = 8
        y = x = image.shape[0] // scale
        r = image.shape[0] // scale - 4

        color_wheel, alpha = get_color_wheel(x, y, r, image)

        # Combine the color wheel and the vector field
        alpha = alpha[:, :, None]
        image = image * (1 - alpha) + color_wheel * alpha

    return image


def get_color_wheel(x, y, r, rgb):
    """
    Create a color wheel
    Args:
        x: x coordinate of the center of the color wheel
        y: y coordinate of the center of the color wheel
        r: radius of the color wheel
        rgb: rgb color of the color wheel
    """
    # Create a color wheel
    color_wheel = np.zeros((rgb.shape[0], rgb.shape[1], 3))
    alpha = np.zeros((rgb.shape[0], rgb.shape[1]))
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            if (i - x)**2 + (j - y)**2 <= r**2:

                angle = np.arctan2(j - y, i - x)
                angle = angle + np.pi
                hue = angle / (2 * np.pi)

                mag = np.sqrt((i - x)**2 + (j - y)**2)
                value = mag / r

                saturation = np.ones_like(hue)

                color_wheel[i, j, :] = np.array([hue, saturation, value])
                # Soften the edges
                alpha[i, j] = np.clip(1 - (mag - r + 1), 0, 1)

    color_wheel = hsv_to_rgb(color_wheel)

    return color_wheel, alpha


def VectorPearsonCorrelation(x, y):
    """
    Calculates the Pearson correlation between two vectors sets
    Args:
        x: (num_samples, num_features)
        y: (num_samples, num_features)
    """
    mean_x = torch.mean(x, dim=0, keepdim=True)
    mean_y = torch.mean(y, dim=0, keepdim=True)

    diff_x = x - mean_x
    diff_y = y - mean_y

    sigma_x = torch.sqrt(torch.sum(diff_x**2))
    sigma_y = torch.sqrt(torch.sum(diff_y**2))

    pcc = torch.sum(diff_x * diff_y) / (sigma_x * sigma_y)
    return pcc


def get_dig_ind(N, offset):
    """Returns the indices of the diagonal elements of the offset diagonal matrices.
    """
    if offset == 0:
        return np.diag_indices(N)
    indices = np.array(np.diag_indices(N - np.abs(offset)))
    if offset > 0:
        indices[1] += offset
    else:
        indices[0] -= offset
    return (indices[0], indices[1])
