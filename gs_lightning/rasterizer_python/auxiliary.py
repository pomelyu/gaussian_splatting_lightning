import torch
from torch import Tensor


def in_frustum(orig_points: Tensor, viewmatrix: Tensor, projmatrix: Tensor) -> Tensor:
    N = len(orig_points)
    assert orig_points.shape == (N, 3)
    assert viewmatrix.shape == (4, 4)
    assert projmatrix.shape == (4, 4)

    p_hom = transformPoint4x4(orig_points, projmatrix)
    p_w = 1. / (p_hom[:, -1:] + 0.0000001)
    p_proj = p_hom * p_w
    p_view = transformPoint4x3(orig_points, viewmatrix)

    # invisible = (p_view[:, 2] <= 0.2) | (p_proj[:, 0] < -1.3) | (p_proj[:, 0] > 1.3) | (p_proj[:, 1] < -1.3) | (p_proj[:, 1] > 1.3)
    invisible = p_view[:, 2] <= 0.2
    return torch.logical_not(invisible)


def transformPoint4x4(points_3d: Tensor, matrix_4x4: Tensor) -> Tensor:
    N = len(points_3d)
    assert points_3d.shape == (N, 3)
    assert matrix_4x4.shape == (4, 4)
    homogeneous_points_3d = torch.cat([points_3d, torch.ones(N, 1).to(points_3d)], -1)
    out = homogeneous_points_3d @ matrix_4x4
    assert out.shape == (N, 4)
    return out

def transformPoint4x3(points_3d: Tensor, matrix_4x3: Tensor) -> Tensor:
    N = len(points_3d)
    assert points_3d.shape == (N, 3)
    assert matrix_4x3.shape == (4, 4)
    out = transformPoint4x4(points_3d, matrix_4x3)
    out = out[:, :3]
    assert out.shape == (N, 3)
    return out
