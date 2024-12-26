from typing import Optional
from typing import Tuple

import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings
from torch import Tensor

from .auxiliary import BLOCK_X
from .auxiliary import BLOCK_Y
from .auxiliary import NUM_CHANNELS
from .auxiliary import SortPairs
from .auxiliary import in_frustum
from .forward import getRect
from .forward import preprocess
from .forward import render


def markVisible(means3D: Tensor, viewmatrix: Tensor, projmatrix: Tensor) -> Tensor:
    return in_frustum(means3D, viewmatrix, projmatrix)

# 1. diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py: GaussianRasterizer.forward
# 2. diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py: rasterize_gaussians
# 3. diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py: _RasterizeGaussians.forward
# 4. diff-gaussian-rasterization/ext.cpp: m.def("rasterize_gaussians", &RasterizeGaussiansCUDA)
# 5. diff-gaussian-rasterization/rasterize_points.cu: RasterizeGaussiansCUDA
# 6. diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu: CudaRasterizer::Rasterizer::forward
def rasterize_gaussians(
    means3D: Tensor,
    means2D: Tensor,
    shs: Tensor,
    opacities: Tensor,
    scales: Tensor,
    rotations: Tensor,
    raster_settings: GaussianRasterizationSettings,
    conv3D_precomp: Optional[Tensor] = None,
    colors_precomp: Optional[Tensor] = None,
):
    height = raster_settings.image_height
    width = raster_settings.image_width

    focal_y = (height * 0.5) / raster_settings.tanfovy
    focal_x = (width * 0.5) / raster_settings.tanfovx

    # the numbers of tiles in each dimension, note the tile size is BLOCK_X * BLOXK_Y(16x16)
    # tile_grid and block are also used as <<<grid, block>>> when triggering a cuda kernel
    tile_grid = ((width + BLOCK_X - 1) // BLOCK_X, (height + BLOCK_Y - 1) // BLOCK_Y, 1)
    block = (BLOCK_X, BLOCK_Y, 1)

    if NUM_CHANNELS != 3 and colors_precomp is None:
        raise RuntimeError("colors_precomp is required for non-RGB color")

    # Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
    rgb, clamped, depths, radii, means2D, conv3D, conic_opacity, tiles_touched = preprocess(
        H=height,
        W=width,
        means3D=means3D,
        opacity=opacities,
        viewmatrix=raster_settings.viewmatrix,
        projmatrix=raster_settings.projmatrix,
        cam_pos=raster_settings.campos,
        tan_fovx=raster_settings.tanfovx,
        tan_fovy=raster_settings.tanfovy,
        focal_x=focal_x,
        focal_y=focal_y,
        tile_grid=tile_grid,
        scale_modifier=raster_settings.scale_modifier,
        sh_degree=raster_settings.sh_degree,
        scales=scales,
        rotations=rotations,
        shs=shs,
        conv3D_precomp=conv3D_precomp,
        colors_precomp=colors_precomp,
        prefiltered=raster_settings.prefiltered,
        antialiasing=raster_settings.antialiasing,
    )

    # Figure out the gaussians that cover the tile and sort them by their depth
    #   1. create "point_list_keys_unsorted", a list that store the covered tile idx and gaussian depth of a gaussian
    #   2. argument sort "point_list_keys_unsorted" by its value
    #   3. use the sorted idx to find the guassian id in depth from "point_list_unsorted"

    # tiles_touched: the number of tiles be touched by a specific gaussian
    # point_offsets: the i-th gaussian store information from point_offsets[i]-th to point_offsets[i+1]-th elements of
    # gaussian_keys_unsorted/point_list_keys_unsorted and gaussian_values_unsorted/point_list_unsorted

	# Compute prefix sum over full list of touched tile counts by Gaussians
	# E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
    point_offsets = torch.cumprod(tiles_touched)
    num_rendered = point_offsets[-1]

    point_list_keys_unsorted, point_list_unsorted = duplicateWithKeys(means2D, radii, depths, point_offsets, tile_grid)
    point_list_keys, point_list = SortPairs(point_list_keys_unsorted, point_list_unsorted)

    ranges = identifyTileRanges(point_list_keys, tile_grid)

    rendered_image, render_depth = render(
        means2D=means2D,
        rgb=rgb,
        depths=depths,
        conic_opacity=conic_opacity,
        ranges=ranges,
        point_list=point_list,
        background=raster_settings.bg,
        tile_grid=tile_grid,
    )
    return rendered_image, radii, render_depth


def duplicateWithKeys(
    means2D: Tensor,
    radii: Tensor,
    depths: Tensor,
    point_offsets: Tensor,
    tile_grid: tuple,
) -> Tuple[Tensor, Tensor]:
    num_rendered = point_offsets[-1]
    point_list_keys_unsorted = torch.zeros(num_rendered, dtype=torch.float64)
    point_list_unsorted = torch.zeros(num_rendered, dtype=torch.int64)

    rect = getRect(means2D, radii, tile_grid)
    for gs_idx in len(means2D):
        if radii[gs_idx] <= 0:
            continue

        offset = 0 if gs_idx == 0 else point_offsets[gs_idx - 1]
        x_min, y_min, x_max, y_max = rect[gs_idx]
        # traverse all the touched tiles of this gaussian splat
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                tile_id = y * tile_grid[0] + x
                depth = depths[gs_idx]
                # key is tileID(32bit integer) | depths(32bit float)
                key = np.int64(tile_id) << 32 | np.float32(depth)
                point_list_keys_unsorted[offset] = key
                point_list_unsorted[offset] = gs_idx
                offset += 1

# Check the start and end idx of a tile in point_list_keys
# the shape of returns Tensor is (num_tiles, 2), indicating start and end idx
def identifyTileRanges(point_list_keys: Tensor, tile_grid: tuple) -> Tensor:
    num_tiles = tile_grid[0] * tile_grid[1] * tile_grid[2]
    ranges = torch.zeros(num_tiles, 2)
    for idx, key in enumerate(point_list_keys):
        tile_id = key >> 32
        if idx == 0:
            ranges[tile_id][0] = 0
            continue

        prev_tile_id = point_list_keys[idx -1] >> 32
        if tile_id != prev_tile_id:
            ranges[prev_tile_id][1] = idx
            ranges[tile_id][0] = idx

        if idx == len(point_list_keys) - 1:
            ranges[tile_id][1] = len(point_list_keys) - 1
    return ranges
