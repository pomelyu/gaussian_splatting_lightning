import torch
from torch import Tensor

__all__ = ["apply_projection_matrix"]

def apply_projection_matrix(points_3d: Tensor, proj_matrix: Tensor, eps: float = 1e-7) -> Tensor:
    return torch.ops.gs_kernels.project2d.default(points_3d, proj_matrix, eps)

@torch.library.register_fake("gs_kernels::project2d")
def _(points_3d: Tensor, proj_matrix: Tensor, eps: float = 1e-7):
    N = len(points_3d)
    torch._check(points_3d.dtype == torch.float)
    torch._check(points_3d.shape == (N, 3))
    torch._check(proj_matrix.dtype == torch.float)
    torch._check(proj_matrix.shape == (4, 4))
    return torch.empty(N, 2)
