from typing import List
from typing import Tuple

import torch
from tqdm import tqdm

from .camera_tools import apply_extrinsic_matrix
from .camera_tools import apply_projection_matrix
from .camera_tools import in_frustum
from .camera_tools import ndc2Pix
from .render_tools import compute_extent_and_radius
from .render_tools import compute_gaussian_weight
from .render_tools import computeColorFromSH
from .render_tools import computeConv2D
from .render_tools import computeConv3D
from .render_tools import get_covered_tiles
from .render_tools import inverse_conv2D

BLOCK_X = 16
BLOCK_Y = 16
NUM_CHANNELS = 3

def rasterize_gaussian(
    # Gaussian splats parameters
    means3D: torch.Tensor,      # (N, 3), gaussian splats position in world coordinates
    opacities: torch.Tensor,    # (N, 1)
    scales: torch.Tensor,       # (N, 3)
    rotations: torch.Tensor,    # (N, 4)
    shs: torch.Tensor,          # (N, d_sh)
    scale_modifier: float,
    # Rasterization parameters
    image_width: int,
    image_height: int,
    tanfovx: float,
    tanfovy: float,
    viewmatrix: torch.Tensor,   # (4, 4), camera extrinsic matrix
    projmatrix: torch.Tensor,   # (4, 4), full projection matrix, i.e. camera extrinsic + intrinsic(projection) matrix
    campos: torch.Tensor,       # (3,), camera position in world coordinates
    background: torch.Tensor,   # (3,), RGB
    sh_degree: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # rendered_image, gaussian radii, depth_image

    N = len(means3D)
    device = means3D.device

    # The visible gaussians to be rendered
    visible_mask = torch.full((N,), False, device=device)

    p_view = apply_extrinsic_matrix(means3D, viewmatrix)
    p_proj = apply_projection_matrix(means3D, projmatrix)

    visible_mask[in_frustum(p_view, p_proj)] = True

    # To perform rasterization, we need to project 3d gaussian to 2d plane
    # Though a 3d gaussian after perspective projection is not a 2d gaussian, we can use 2d gaussian to approximate it
    # see EWA Splatting and https://math.stackexchange.com/questions/4716499/pinhole-camera-projection-of-3d-multivariate-gaussian
    # also https://towardsdatascience.com/a-python-engineers-introduction-to-3d-gaussian-splatting-part-2-7e45b270c1df
    focal_x = (image_width * 0.5) / tanfovx
    focal_y = (image_height * 0.5) / tanfovy
    conv3D = computeConv3D(scales, scale_modifier, rotations)
    conv2D = computeConv2D(means3D, focal_x, focal_y, tanfovx, tanfovy, conv3D, viewmatrix)

    # Next, we need to compute the inversed matrix of covarinace_2d
    # so we can get the pixel strength from G = exp((x-mu)E^-1(x-mu)^T)
    inv_conv2D, invalid_mask, h_convolution_scaling = inverse_conv2D(conv2D)
    visible_mask[invalid_mask] = False

    # To find which gaussian contributes to a pixel, we selects gaussians which cover the pixel
    # As a result, we have to calculate the 3 standard deviations of every gaussian, wich cover 99.7% of the distribution
    # Note that the eigen-values of a matrix are the sqaure of major and minor axis lengths of the ellipse
    # and also the varaince of the corresponding gaussian
    radius = compute_extent_and_radius(conv2D)
    radius = torch.where(visible_mask, radius, 0)
    depths = p_view[:, 2]

    color = computeColorFromSH(sh_degree, means3D, campos, shs)

    # Divide image into several tiles and render them parallelly

    # numbers of tile(block) in each dimension
    grid = ((image_width + BLOCK_X - 1) // BLOCK_X, (image_height + BLOCK_Y - 1) // BLOCK_Y, 1)
    # block(tile) size
    block = (BLOCK_X, BLOCK_Y, 1)

    # calculate the tiles that covered by each gaussian
    p_image = ndc2Pix(p_proj, image_width, image_height)
    gaussian_in_tiles = calculate_gaussian_in_tiles(p_image, radius, grid, block)

    rect = get_covered_tiles(p_image, radius, grid, block)
    invalid_mask = (rect[:, 2] - rect[:, 0]) * (rect[:, 3] - rect[:, 1]) == 0
    visible_mask[invalid_mask] = False
    radius[invalid_mask] = 0

    canvas = torch.zeros(image_height, image_width, NUM_CHANNELS).to(device)
    depth_canvas = torch.zeros(image_height, image_width).to(device)

    # for x, y in p_image[visible_mask]:
    #     x, y = int(x), int(y)
    #     if 0 <= x < image_width and 0 <= y < image_height:
    #         canvas[y, x] = 1.0
    # canvas = torch.moveaxis(canvas, -1, 0)
    # depth_canvas = depth_canvas.unsqueeze(0)
    # return canvas, radius, depth_canvas

    pbar = tqdm(total=grid[0]*grid[1], desc="rendering pixels", unit="step")
    for y in range(grid[1]):
        for x in range(grid[0]):
            tile_id = y * grid[0] + x
            offset_x = x * block[0]
            offset_y = y * block[1]
            render_tile(
                canvas, depth_canvas, offset_x, offset_y, gaussian_in_tiles[tile_id],
                p_image, color, opacities, depths, inv_conv2D, background, block)
            pbar.update(1)
    pbar.close()

    canvas = torch.moveaxis(canvas, -1, 0)
    depth_canvas = depth_canvas.unsqueeze(0)
    return canvas, radius, depth_canvas

def calculate_gaussian_in_tiles(
    point_images: torch.Tensor,
    radius: torch.Tensor,
    grid: tuple,
    block: tuple,
) -> torch.Tensor:
    gaussian_in_tiles = [[] for _ in range(grid[0] * grid[1])]
    rect = get_covered_tiles(point_images, radius, grid, block)
    for i, (x_min, y_min, x_max, y_max) in tqdm(enumerate(rect), total=len(rect)):
        if (x_max - x_min) * (y_max - y_min) == 0:
            continue
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                tile_id = y * grid[0] + x
                gaussian_in_tiles[tile_id].append(i)
    return gaussian_in_tiles

def render_tile(
    canvas: torch.Tensor,
    depth_canvas: torch.Tensor,
    offset_x: int,
    offset_y: int,
    gs_in_tile: List[int],
    p_image: torch.Tensor,
    color: torch.Tensor,
    opacity: torch.Tensor,
    depth: torch.Tensor,
    inv_conv2D: torch.Tensor,
    background: torch.Tensor,
    block: tuple,
):
    H, W = canvas.shape[:2]
    # sort gaussians by their depth value
    depth = depth[gs_in_tile]
    sorted_idx = torch.argsort(depth)
    p_image = p_image[sorted_idx]
    color = color[sorted_idx]
    opacity = opacity[sorted_idx]
    depth = depth[sorted_idx]
    inv_conv2D = inv_conv2D[sorted_idx]
    for y in range(offset_y, min(offset_y + block[1], H)):
        for x in range(offset_x, min(offset_x + block[0], W)):
            render_pixel(canvas, depth_canvas, x, y, p_image, color, opacity, depth, inv_conv2D, background)

def render_pixel(
    canvas: torch.Tensor,
    depth_canvas: torch.Tensor,
    x: int,
    y: int,
    p_image: torch.Tensor,
    color: torch.Tensor,
    opacity: torch.Tensor,
    depth: torch.Tensor,
    inv_conv2D: torch.Tensor,
    background: torch.Tensor,
    threshold: float = 1. / 255
) -> None:
    coord = torch.Tensor([[x, y]]).to(p_image.device)
    weight = compute_gaussian_weight(coord, p_image, inv_conv2D)
    alpha = torch.clamp_max(weight * opacity.squeeze(-1), 0.99)

    remain_weight = 1
    for a, rgb, d in zip(alpha, color, depth):
        if a < threshold:
            continue

        test_weight = remain_weight * (1 - a)
        if test_weight < 0.0001:
            return
        
        canvas[y, x] += rgb * a * remain_weight
        depth_canvas[y, x] += (1 / d) * a * remain_weight
        remain_weight = test_weight

    canvas[y, x] += remain_weight * background
