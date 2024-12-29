from typing import Tuple

import kornia as K
import torch
from torch import Tensor

from gs_lightning.utils.sh import apply_sh

from .auxiliary import BLOCK_X
from .auxiliary import BLOCK_Y
from .auxiliary import NUM_CHANNELS
from .auxiliary import getRect
from .auxiliary import in_frustum
from .auxiliary import ndc2Pix
from .auxiliary import transformPoint4x3
from .auxiliary import transformPoint4x4


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
    N = len(points)
    points = transformPoint4x3(mean3d, viewmatrix)

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
    L = R @ S
    covar = L @ L.transpose(1, 2)                   # (N, 3, 3)
    # use the low diagnoal elements to make sure the matrix is symmetric
    out = torch.stack([
        covar[:, 0, 0], covar[:, 0, 1], covar[:, 0, 2],
        covar[:, 1, 1], covar[:, 1, 2], covar[:, 2, 2],
    ], -1)
    return out

# Perform initial steps for each Gaussian prior to rasterization.
def preprocess(
    H: int,
    W: int,
    means3D: Tensor,        # (N, 3)
    opacity: Tensor,        # (N, 1)
    viewmatrix: Tensor,     # (N, 4, 4)
    projmatrix: Tensor,     # (N, 4, 4)
    cam_pos: Tensor,        # (N, 3)
    tan_fovx: float,
    tan_fovy: float,
    focal_x: float,
    focal_y: float,
    tile_grid: tuple,
    scale_modifier: float,
    sh_degree: int,                 # =D
    scales: Tensor = None,          # (N, 3)
    rotations: Tensor = None,       # (N, 4)
    shs: Tensor = None,
    conv3D_precomp: Tensor = None,  # (N, 6) or (N, 3, 3)
    colors_precomp: Tensor = None,  # (N, 3)
    prefiltered: bool = False,
    antialiasing: bool = False,
):
    N = len(means3D)
    device = means3D.device

	# Initialize radius and touched tiles to 0. If this isn't changed,
    rgb = torch.zeros(N, 3).to(device)
    depths = torch.zeros(N).to(device)
    radii = torch.zeros(N).to(device)
    points_xy_image = torch.zeros(N, 2).to(device)
    conic_opacity = torch.zeros(N, 4).to(device)
    tiles_touched = torch.zeros(N).to(device)

    # find the points located inside the view frustum
    # TODO: use filtered
    visible_mask = in_frustum(means3D, viewmatrix, projmatrix)
    valid_idx = torch.arange(N).long().to(device)[visible_mask]

    # transform visible point by projecting
    p_hom = transformPoint4x4(means3D[visible_mask], projmatrix)
    p_w = 1. / (p_hom[:, -1:] + 0.0000001)
    p_proj = p_hom * p_w
    p_view = transformPoint4x3(means3D[visible_mask], viewmatrix)

	# If 3D covariance matrix is precomputed, use it, otherwise compute
	# from scaling and rotation parameters. 
    if conv3D_precomp is not None:
        conv3D = conv3D_precomp
    else:
        conv3D = computeConv3D(scales, scale_modifier, rotations)

    # To perform rasterization, we need to project 3d gaussian to 2d plane
    # Though a 3d gaussian after perspective projection is not a 2d gaussian, we can use 2d gaussian to approximate it
    # see EWA Splatting and https://math.stackexchange.com/questions/4716499/pinhole-camera-projection-of-3d-multivariate-gaussian
    # also https://towardsdatascience.com/a-python-engineers-introduction-to-3d-gaussian-splatting-part-2-7e45b270c1df
    conv2D = computeConv2D(means3D[visible_mask], focal_x, focal_y, tan_fovx, tan_fovy, conv3D[visible_mask], viewmatrix)

    # Next, we need to compute the inversed matrix of covarinace_2d
    # so we can get the pixel strength from G = exp((x-mu)E^-1(x-mu)^T)
    conic, h_convolution_scaling, valid_conv2D_mask = inverse_conv2D(conv2D, antialiasing)
    valid_idx = valid_idx[valid_conv2D_mask]

    # To find which gaussian contributes to a pixel, we selects gaussians which cover the pixel
    # As a result, we have to calculate the 3 standard deviations of every gaussian, wich cover 99.7% of the distribution
    # Note that the eigen-values of a matrix are the sqaure of major and minor axis lengths of the ellipse
    # and also the varaince of the corresponding gaussian
    radius = compute_extent_and_radius(conv2D)

    point_image = torch.stack([ndc2Pix(p_proj[valid_conv2D_mask, 0], W), ndc2Pix(p_proj[valid_conv2D_mask, 1], H)], -1)
    # Get the grids that are influenced by the gaussians
    rect = getRect(point_image, radius, tile_grid)
    valid_rect_mask = torch.logical_not((rect[:, 2] - rect[:, 0]) * (rect[:, 3] - rect[:, 1]) == 0)
    valid_idx = valid_idx[valid_rect_mask]

    radii[valid_idx] = radius[valid_rect_mask]
    depths[valid_idx] = p_view[:, 2][valid_conv2D_mask][valid_rect_mask]
    points_xy_image[valid_idx] = point_image[valid_rect_mask]
    conic_opacity[valid_idx, 0] = conic[valid_rect_mask, 0, 0]
    conic_opacity[valid_idx, 1] = conic[valid_rect_mask, 0, 1]
    conic_opacity[valid_idx, 2] = conic[valid_rect_mask, 1, 1]
    conic_opacity[valid_idx, 3] = (opacity[visible_mask][valid_conv2D_mask] * h_convolution_scaling)[valid_rect_mask]
    tiles_touched[valid_idx] = (rect[:, 2] - rect[:, 0]) * (rect[:, 3] - rect[:, 1])[valid_rect_mask]

    if colors_precomp is None:
        rgb, clamped = computeColorFromSH(sh_degree, means3D, cam_pos, shs)
    else:
        rgb = clamped = None

    return rgb, clamped, depths, radii, points_xy_image, conv3D, conic_opacity, tiles_touched

def inverse_conv2D(conv2D: Tensor, antialiasing: bool) -> Tuple[Tensor, Tensor, Tensor]:
    N = len(conv2D)
    assert conv2D.shape == (N, 2, 2)

    det_cov = conv2D[:, 0, 0] * conv2D[:, 1, 1] - conv2D[:, 0, 1] * conv2D[:, 0, 1]

    h_var = 0.3     # FIXME: why
    conv2D[:, 0, 0] += h_var
    conv2D[:, 1, 1] += h_var
    det_cov_plus_h_cov = conv2D[:, 0, 0] * conv2D[:, 1, 1] - conv2D[:, 0, 1] * conv2D[:, 0, 1]
    det = det_cov_plus_h_cov

    valid_mask = torch.logical_not(det == 0)
    det_inv = 1. / det[valid_mask]
    # FIXME: what does "conic" mean, why not using inv_conv2D
    conic = torch.cat([
        conv2D[valid_mask, 1, 1] * det_inv, -conv2D[valid_mask, 0, 1] * det_inv,
        -conv2D[valid_mask, 0, 1] * det_inv, conv2D[valid_mask, 0, 0] * det_inv,
    ]).reshape(-1, 2, 2)

    if antialiasing:
        h_convolution_scaling = torch.sqrt(torch.clamp_min(det_cov / det_cov_plus_h_cov, 0.000025))
    else:
        h_convolution_scaling = torch.ones_like(det_cov)
    h_convolution_scaling = h_convolution_scaling[valid_mask]

    return conic, h_convolution_scaling, valid_mask

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

def render(
    mean2D: Tensor,
    rgb: Tensor,
    depths: Tensor,
    conic_opacity: Tensor,
    ranges: Tensor,
    point_list: Tensor,
    background: Tensor,
    tile_grid: tuple,
) -> Tensor:
    num_tiles = tile_grid[0] * tile_grid[1] * tile_grid[2]
    rendered_tiles = []
    rendered_depth_invs = []
    for tile_id in range(num_tiles):
        tile_color = torch.zeros(BLOCK_Y, BLOCK_X, NUM_CHANNELS)
        tile_weight = torch.ones(BLOCK_Y, BLOCK_X)
        tile_depth_inv = torch.zeros(BLOCK_Y, BLOCK_X)
        for pixel_y in range(BLOCK_Y):
            for pixel_x in range(BLOCK_X):
                done = False
                for gs_id in point_list[ranges[tile_id, 0], ranges[tile_id, 1]]:
                    tile_offset_x = (tile_id % tile_grid[0]) * BLOCK_X
                    tile_offset_y = (tile_id // tile_grid[0]) * BLOCK_Y
                    pixel_coord = [tile_offset_x + pixel_x, tile_offset_y + pixel_y]
                    weight = computeGaussianWeight(pixel_coord, mean2D[gs_id], conic_opacity[gs_id, :3])
                    alpha = torch.clamp_max(weight * conic_opacity[gs_id, -1], 0.99)
                    if alpha < 1. / 255.:
                        continue

                    test_weight = tile_weight[pixel_y, pixel_x] * (1 - alpha)
                    if test_weight < 0.0001:
                        done = True

                    tile_color[pixel_y, pixel_x] += rgb[gs_id] * alpha * tile_weight[pixel_y, pixel_x]
                    tile_depth_inv += (1 / depths[gs_id]) * alpha * tile_weight[pixel_y, pixel_x]
                    tile_weight[pixel_y, pixel_x] = test_weight

                    if done:
                        break

                tile_color[pixel_y, pixel_x] += tile_weight[pixel_y, pixel_x] * background[:, :, None]

        rendered_tiles.append(tile_color)
        rendered_depth_invs.append(tile_depth_inv)

    rendered_tiles = torch.stack(rendered_tiles).reshape(tile_grid[1], tile_grid[0], BLOCK_Y, BLOCK_X, NUM_CHANNELS)
    rendered_tiles = torch.cat(torch.cat(rendered_tiles, 1), 1)

    rendered_depth_invs = torch.stack(rendered_depth_invs).reshape(tile_grid[1], tile_grid[0], BLOCK_Y, BLOCK_X)
    rendered_depth_invs = torch.cat(torch.cat(rendered_depth_invs, 1), 1)

    return rendered_tiles, rendered_depth_invs


def computeGaussianWeight(pixel_coord, gaussian_mean, conic):
    diff = pixel_coord - gaussian_mean
    x, y = diff
    c0, c1, c2 = conic
    # power = -0.5 * diff @ inversed_covariance @ diff.T
    power = -0.5 * (c0 * x * x + c1 * y * y - 2 * c2 * x * y)
    return torch.exp(torch.clamp(power, 0))


def computeColorFromSH(sh_degree: int, points: Tensor, cam_pos: Tensor, sh_params: Tensor) -> torch.Tensor:
    N = len(points)
    assert points.shape == (N, 3)
    assert cam_pos.shape == (3,)
    assert len(sh_params) == N and sh_params.ndim == 2

    direction = points - cam_pos[None, :]
    direction = direction / torch.linalg.norm(direction, dim=-1, keepdim=True)

    results = apply_sh(sh_params, direction, sh_degree)
    results += 0.5

	# RGB colors are clamped to positive values. If values are
	# clamped, we need to keep track of this for the backward pass.
    clamped = torch.stack([
        results[:, 0] < 0,
        results[:, 1] < 0,
        results[:, 2] < 0,
    ], dim=-1)
    results = torch.clamp_min(results, 0.0)
    return results, clamped

