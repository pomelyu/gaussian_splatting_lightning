import numpy as np


def get_projection_matrix(
    fx: float,
    fy: float,
    w: int,
    h: int,
    znear: float,
    zfar: float,
) -> np.ndarray:
    """
    Perspective Matrix. Note that this matrix is specified for colum vector, i.e. m @ x
    https://github.com/graphdeco-inria/gaussian-splatting/issues/826
    Unlike the projection matrix used in OpenGL, this matrix project z to [0, 1] instead of [-1, 1] and has different sign in z axis
    https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/opengl-perspective-projection-matrix.html
    [
        [2*n/(r-l),         0, (r+l)/(r-l),          0],
        [        0, 2*n/(t-b), (t+b)/(t-b),          0],
        [        0,         0, (f+n)/(f-n), -f*n/(f-n)],
        [        0,         0,           1 ,         0],
    ]
    """

    # calculate top-bottom, left-right cliping plane
    right = (w * 0.5) * (znear / fx)
    left = -right
    top = (h * 0.5) * (znear / fy)
    bottom = -top

    m = np.zeros((4, 4))
    zsign = 1.0

    m[0, 0] = (2 * znear) / (right - left)
    m[1, 1] = (2 * znear) / (top - bottom)
    # m[0, 2] = (right + left) / (right - left) # = 0
    # m[1, 2] = (top + bottom) / (top - bottom) # = 0
    m[3, 2] = zsign
    m[2, 2] = zsign * (zfar + znear) / (zfar - znear)
    m[2, 3] = -(zfar * znear) / (zfar - znear)
    return m
