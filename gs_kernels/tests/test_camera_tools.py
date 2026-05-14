import torch
from torch import Tensor
from torch.testing._internal.common_utils import TestCase


def reference_project2d(points_3d: Tensor, proj_matrix: Tensor, eps: float = 1e-7) -> Tensor:
    N = len(points_3d)
    assert points_3d.shape == (N, 3)
    assert proj_matrix.shape == (4, 4)

    p_hom = torch.cat([points_3d, torch.ones(N, 1).to(points_3d)], -1)
    p_hom = p_hom @ proj_matrix
    p_w = 1. / (p_hom[:, -1:] + eps)
    p_proj = p_hom * p_w
    return p_proj[:, :2]

def create_random_project_matrix(device='cpu', requires_grad=False) -> Tensor:
    fx = torch.rand(1).item() * 900 + 100   # focal length x in [100, 1000]
    fy = torch.rand(1).item() * 900 + 100   # focal length y in [100, 1000]
    w = torch.randint(256, 1920, (1,)).item()
    h = torch.randint(256, 1080, (1,)).item()
    znear, zfar = 0.01, 100.0

    right = (w * 0.5) * (znear / fx)
    left = -right
    top = (h * 0.5) * (znear / fy)
    bottom = -top

    # Build column-vector projection matrix (same convention as get_projection_matrix)
    # then transpose for row-vector use: p_hom @ proj_matrix
    m = torch.zeros(4, 4)
    m[0, 0] = (2 * znear) / (right - left)
    m[1, 1] = (2 * znear) / (top - bottom)
    m[2, 2] = (zfar + znear) / (zfar - znear)
    m[2, 3] = -(zfar * znear) / (zfar - znear)
    m[3, 2] = 1.0

    m = m.T.to(dtype=torch.float32, device=device)
    m.requires_grad_(requires_grad)
    return m

class TestApplyProjectMatrix(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(*size):
            return torch.randn(size, device=device, requires_grad=requires_grad)

        def make_nodiff_tensor(*size):
            return torch.randn(size, device=device, requires_grad=False)

        return [
            [make_tensor(1000, 3), create_random_project_matrix(device, requires_grad=requires_grad), 1e-7],
            [make_tensor(1000, 3), create_random_project_matrix(device, requires_grad=False), 1e-7],
            [make_nodiff_tensor(1000, 3), create_random_project_matrix(device), 1e-7],
        ]

    def test_correctness_cpu(self):
        self._test_correctness("cpu")

    def test_correctness_cuda(self):
        self._test_correctness("cuda")

    def _test_correctness(self, device):
        import gs_kernels
        samples = self.sample_inputs(device)
        for args in samples:
            result = gs_kernels.apply_projection_matrix(*args)
            expected = reference_project2d(*args)
            torch.testing.assert_close(result, expected)

