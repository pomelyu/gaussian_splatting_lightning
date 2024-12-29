import torch
from torch import Tensor


def in_frustum(transformed_points_3d: Tensor, projected_points_2d: Tensor) -> Tensor:
    # invisible = (p_view[:, 2] <= 0.2) | (p_proj[:, 0] < -1.3) | (p_proj[:, 0] > 1.3) | (p_proj[:, 1] < -1.3) | (p_proj[:, 1] > 1.3)
    visible_mask = transformed_points_3d[:, 2] > 0.2
    return visible_mask

def apply_projection_matrix(points_3d: Tensor, proj_matrix: Tensor, eps: float = 1e-7) -> Tensor:
    N = len(points_3d)
    assert points_3d.shape == (N, 3)
    assert proj_matrix.shape == (4, 4)

    p_hom = torch.cat([points_3d, torch.ones(N, 1).to(points_3d)], -1)
    p_hom = p_hom @ proj_matrix
    p_w = 1. / (p_hom[:, -1:] + eps)
    p_proj = p_hom * p_w
    return p_proj[:, :2]

def apply_extrinsic_matrix(points_3d: Tensor, extrinsic_matrix: Tensor) -> Tensor:
    N = len(points_3d)
    assert points_3d.shape == (N, 3)
    assert extrinsic_matrix.shape == (4, 4)
    homogeneous_points_3d = torch.cat([points_3d, torch.ones(N, 1).to(points_3d)], -1)
    out = homogeneous_points_3d @ extrinsic_matrix
    return out

def ndc2Pix(v: Tensor, w: int, h: int) -> Tensor:
    x = ((v[:, 0] + 1.0) * w - 1.0) * 0.5
    y = ((v[:, 1] + 1.0) * h - 1.0) * 0.5
    return torch.stack([x, y], -1)
