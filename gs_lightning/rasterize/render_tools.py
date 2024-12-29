from typing import Tuple

import kornia as K
import torch
from torch import Tensor

from gs_lightning.utils.sh import apply_sh

from .camera_tools import apply_extrinsic_matrix


# Compute 2D screen-space covariance matrix
def computeConv2D(
    mean3d: Tensor,     # (N, 3)
    focal_x: float,
    focal_y: float,
    tan_fovx: float,
    tan_fovy: float,
    conv3D: Tensor,     # (N, 3, 3) or (N, 6)
    viewmatrix: Tensor, # (4, 4)
):
    N = len(mean3d)
    points = apply_extrinsic_matrix(mean3d, viewmatrix)

    z = points[:, 2]
    lim_x = 1.3 * tan_fovx * z
    lim_y = 1.3 * tan_fovy * z
    x = torch.clamp(points[:, 0], -lim_x, lim_x)
    y = torch.clamp(points[:, 1], -lim_y, lim_y)

    J = torch.zeros(N, 3, 3).to(points)
    J[:, 0, 0] = focal_x / z
    J[:, 0, 2] = -(focal_x * x) / (z**2)
    J[:, 1, 1] = focal_y / z
    J[:, 1, 2] = -(focal_y * y) / (z**2)

    W = viewmatrix[None, :3, :3].expand(N, -1, -1)
    T = W @ J

    if conv3D.shape == (N, 6):
        conv3D = torch.stack([
            conv3D[:, 0], conv3D[:, 1], conv3D[:, 2],
            conv3D[:, 1], conv3D[:, 3], conv3D[:, 4],
            conv3D[:, 2], conv3D[:, 4], conv3D[:, 5],
        ], dim=-1).reshape(N, 3, 3)
    assert conv3D.shape == (N, 3, 3)

    conv2D = T.transpose(1, 2) @ conv3D.transpose(1, 2) @ T
    conv2D = conv2D[:, :2, :2]
    return conv2D

# calculate covariance matrix from scaling and rotation
# eq6. Covariance = R @ S @ S^T @ R^T
def computeConv3D(
    scale: Tensor,          # (N, 3)
    scale_modifier: float,
    rotation: Tensor,       # (N, 4)
):
    R = K.geometry.Quaternion(rotation).matrix()    # (N, 3, 3)
    S = torch.diag_embed(scale * scale_modifier)    # (N, 3, 3)
    L = S @ R
    covar = L.transpose(1, 2) @ L                   # (N, 3, 3)
    # use the low diagnoal elements to make sure the matrix is symmetric
    out = torch.stack([
        covar[:, 0, 0], covar[:, 0, 1], covar[:, 0, 2],
        covar[:, 1, 1], covar[:, 1, 2], covar[:, 2, 2],
    ], -1)
    return out


def inverse_conv2D(conv2D: Tensor, h_var: float = 0.3, antialias: bool = False) -> Tuple[Tensor, Tensor]:
    N = len(conv2D)
    assert conv2D.shape == (N, 2, 2)

    det = conv2D[:, 0, 0] * conv2D[:, 1, 1] - conv2D[:, 0, 1] * conv2D[:, 0, 1]

    conv2D[:, 0, 0] += h_var
    conv2D[:, 1, 1] += h_var
    det_cov_plus_h_cov = conv2D[:, 0, 0] * conv2D[:, 1, 1] - conv2D[:, 0, 1] * conv2D[:, 0, 1]

    if antialias:
        h_convolution_scaling = torch.sqrt(torch.clamp_min(det / det_cov_plus_h_cov, 0.000025))
    else:
        h_convolution_scaling = torch.ones_like(det)
    det = det_cov_plus_h_cov

    invalid_mask = (det == 0)
    det_inv = 1. / torch.clamp_min(det, 1e-5)

    inv_conv2D = torch.cat([
        conv2D[:, 1, 1] * det_inv, -conv2D[:, 0, 1] * det_inv,
        -conv2D[:, 1, 0] * det_inv, conv2D[:, 0, 0] * det_inv,
    ]).reshape(-1, 2, 2)
    return inv_conv2D, invalid_mask, h_convolution_scaling

def compute_extent_and_radius(conv2D: Tensor, radius_factor=3.0, eps=1e-1) -> Tensor:
    """ Calculate eigen-value of the covariance matrix
    For [ [a, b]      eigen value is (a+d)/2 +- sqrt(((a+d)/2)**2 - (ad - bc))
          [c, d] ]
    """
    N = len(conv2D)
    assert conv2D.shape == (N, 2, 2)

    mean_term = 0.5 * (conv2D[:, 0, 0] + conv2D[:, 1, 1])
    det = conv2D[:, 0, 0] * conv2D[:, 1, 1] - conv2D[:, 0, 1] * conv2D[:, 1, 0]
    sqrt_term = mean_term ** 2 - det
    sqrt_term = torch.sqrt(torch.clamp_min(sqrt_term, eps))

    # FIXME: sqrt_term is always positive, so lambda1 > lambda2(?)
    lambda1 = mean_term + sqrt_term
    lambda2 = mean_term - sqrt_term
    std = torch.sqrt(torch.maximum(lambda1, lambda2))
    radius = torch.ceil(radius_factor * std)
    return radius

def computeColorFromSH(sh_degree: int, points: Tensor, campos: Tensor, sh_params: Tensor) -> torch.Tensor:
    N = len(points)
    assert points.shape == (N, 3)
    assert campos.shape == (3,)
    assert len(sh_params) == N and sh_params.ndim == 3

    direction = points - campos[None, :]
    direction = direction / torch.linalg.norm(direction, dim=-1, keepdim=True)

    results = apply_sh(sh_params, direction, sh_degree)
    results += 0.5

    results = torch.clamp_min(results, 0.0)
    return results

# Get the grids that are influenced by the gaussians
def get_covered_tiles(p: Tensor, max_radius: Tensor, grid: tuple, block: tuple) -> Tensor:
    x_min = torch.clamp(((p[:, 0] - max_radius) / block[0]).int(), 0, grid[0])
    y_min = torch.clamp(((p[:, 1] - max_radius) / block[1]).int(), 0, grid[1])
    x_max = torch.clamp(((p[:, 0] + max_radius + block[0] - 1) / block[0]).int(), 0, grid[0])
    y_max = torch.clamp(((p[:, 1] + max_radius + block[1] - 1) / block[1]).int(), 0, grid[1])
    return torch.stack([x_min, y_min, x_max, y_max], -1)

def compute_gaussian_weight(pixel_coord: Tensor, gaussian_mean: Tensor, inv_conv2D: Tensor) -> Tensor:
    diff = (gaussian_mean - pixel_coord)[:, None, :]    # (N, 1, 2)
    power = -0.5 * diff @ inv_conv2D @ diff.transpose(1, 2)
    return torch.exp(power).squeeze(-1).squeeze(-1)
