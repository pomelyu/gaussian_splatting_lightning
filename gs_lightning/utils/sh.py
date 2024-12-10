# https://github.com/sxyu/svox2/blob/ee80e2c4df8f29a407fda5729a494be94ccf9234/svox2/utils.py#L82
import torch

__all__ = ["apply_sh", "rgb2sh0", "sh02rgb"]

# SH coefficients
C0 = 0.28209479177387814
C1 = [
    -0.4886025119029199,
    0.4886025119029199,
    -0.4886025119029199,
]
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

def apply_sh(sh_params: torch.Tensor, directions: torch.Tensor, sh_degree: int) -> torch.Tensor:
    N = len(sh_params)
    assert 0 <= sh_degree <= 4
    assert directions.shape == (N, 3)
    x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]

    result = C0 * sh_params[..., 0]
    if sh_degree < 1:
        return result

    result += (
        C1[0] * sh_params[..., 1] +
        C1[1] * sh_params[..., 2] +
        C1[2] * sh_params[..., 3]
    )

    if sh_degree < 2:
        return result

    xx, yy, zz = x * x, y * y, z * z
    xy, yz, zx = x * y, y * z, z * x
    result += (
        C2[0] * xy * sh_params[..., 4] +
        C2[1] * yz * sh_params[..., 5] +
        C2[2] * (2.0 * zz - xx - yy) * sh_params[..., 6] +
        C2[3] * zx * sh_params[..., 7] +
        C2[4] * (xx - yy) * sh_params[..., 8]
    )

    if sh_degree < 3:
        return result

    result += (
        C3[0] * y * (3 * xx - yy) * sh_params[..., 9] +
        C3[1] * xy * z * sh_params[..., 10] +
        C3[2] * y * (4 * zz - xx - yy)* sh_params[..., 11] +
        C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh_params[..., 12] +
        C3[4] * x * (4 * zz - xx - yy) * sh_params[..., 13] +
        C3[5] * z * (xx - yy) * sh_params[..., 14] +
        C3[6] * x * (xx - 3 * yy) * sh_params[..., 15]
    )

    if sh_degree < 3:
        return result
    
    result += (
        C4[0] * xy * (xx - yy) * sh_params[..., 16] +
        C4[1] * yz * (3 * xx - yy) * sh_params[..., 17] +
        C4[2] * xy * (7 * zz - 1) * sh_params[..., 18] +
        C4[3] * yz * (7 * zz - 3) * sh_params[..., 19] +
        C4[4] * (zz * (35 * zz - 30) + 3) * sh_params[..., 20] +
        C4[5] * zx * (7 * zz - 3) * sh_params[..., 21] +
        C4[6] * (xx - yy) * (7 * zz - 1) * sh_params[..., 22] +
        C4[7] * zx * (xx - 3 * yy) * sh_params[..., 23] +
        C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh_params[..., 24]
    )

    return result

def rgb2sh0(rgb):
    return (rgb - 0.5) / C0

def sh02rgb(sh0):
    return sh0 * C0 + 0.5
