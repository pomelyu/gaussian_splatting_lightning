import torch
from torch import Tensor

from .auxiliary import in_frustum


def markVisible(means3D: Tensor, viewmatrix: Tensor, projmatrix: Tensor) -> Tensor:
    return in_frustum(means3D, viewmatrix, projmatrix)
