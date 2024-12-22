import torch
from diff_gaussian_rasterization import GaussianRasterizer

from gs_lightning.rasterizer_python import markVisible

from .test_cases import points_3d
from .test_cases import settings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_markVisible():
    for setting in settings:
        m = GaussianRasterizer(setting)
        gt_results = m.markVisible(points_3d)
        my_results = markVisible(
            points_3d,
            viewmatrix=setting.viewmatrix,
            projmatrix=setting.projmatrix,
        )
        print("visible ratio", torch.sum(gt_results) / len(gt_results))
        assert torch.all(torch.logical_not(torch.logical_xor(gt_results, my_results)))

if __name__ == "__main__":
    test_markVisible()
