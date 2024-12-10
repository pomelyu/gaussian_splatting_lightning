import torch
from scipy.spatial import KDTree


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1 - x))

# https://github.com/graphdeco-inria/gaussian-splatting/issues/292#issuecomment-2007934451
def distCUDA2(points: torch.Tensor) -> torch.Tensor:
    points_np = points.detach().cpu().float().numpy()
    dists, inds = KDTree(points_np).query(points_np, k=4)
    meanDists = (dists[:, 1:] ** 2).mean(1)

    return torch.tensor(meanDists, dtype=points.dtype, device=points.device)
