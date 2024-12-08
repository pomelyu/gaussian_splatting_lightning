import argparse
from pathlib import Path

import cv2
import imageio as iio
import numpy as np
import pycolmap
import torch
import torchvision
from diff_gaussian_rasterization import GaussianRasterizationSettings
from diff_gaussian_rasterization import GaussianRasterizer
from pycolmap import MapCameraIdToCamera
from pycolmap import MapImageIdToImage

from gs_lightning.third_party.gaussian_splatting.scene import GaussianModel


def get_projection_matrix(
    fx: float,
    fy: float,
    w: int,
    h: int,
    znear: float,
    zfar: float,
):
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
    return torch.Tensor(m)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="path to training checkpoint(.ply)")
    parser.add_argument("--colmap", type=str, help="colmap project used in training", default="data/treehill/sparse/0")
    parser.add_argument("--image", type=str, help="colmap images used in training", default="data/treehill/images")
    parser.add_argument("--down_scale", type=int, help="down scale image", default=1)
    parser.add_argument("--frame", "-n", type=int, default=10, help="frame number")
    parser.add_argument("--output", "-o", type=str, default="results")
    # Model Params
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--white_background", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True, parents=True)

    gaussians = GaussianModel(sh_degree=args.sh_degree)
    gaussians.load_ply(args.model)
    # TODO: what is gaussian.create_from_pcd

    background = [1,1,1] if args.white_background else [0, 0, 0]
    background = torch.Tensor(background).to(device)

    # pipeline_params = Namespace(
    #     convert_SHs_python=False,     # TODO: meaning?
    #     compute_cov3D_python=False,   # TODO: meaning?
    #     debug=False,
    #     antialiasing=False,
    # )
    # rendering = render(camera, gaussians, pipeline_params, background, use_trained_exp=False, separate_sh=False)["render"]

    reconstruction = pycolmap.Reconstruction(args.colmap)
    # {1: Camera(camera_id=1, model=PINHOLE, width=5068, height=3326, params=[4219.170711, 4205.602294, 2534.000000, 1663.000000] (fx, fy, cx, cy))}
    cameras: MapCameraIdToCamera = reconstruction.cameras
    # {1: Image(image_id=1, camera_id=1, name="_DSC8874.JPG", triangulated=657/9322)}
    images: MapImageIdToImage = reconstruction.images

    fid = args.frame
    cid = images[fid].camera_id
    image_path = images[fid].name

    scale = 1 / args.down_scale
    image = iio.v3.imread(Path(args.image) / image_path)
    H, W = image.shape[:2]
    # H, W = int(H*scale), int(W*scale)
    H, W = 2400, 1600
    image = cv2.resize(image, (W, H))

    device = "cuda"
    zfar = 100.0
    znear = 0.01
    camera = cameras[cid]
    world_view_transform = np.eye(4)
    world_view_transform[:, :3] = images[fid].cam_from_world.matrix().T
    world_view_transform = torch.Tensor(world_view_transform).to(device)
    camera_center = world_view_transform.inverse()[3, :3].to(device)
    projection_matrix = get_projection_matrix(camera.focal_length_x, camera.focal_length_y, camera.width, camera.height, znear, zfar).T
    projection_matrix = projection_matrix.to(device)
    full_proj_transform = world_view_transform @ projection_matrix

    raster_settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=(camera.width * 0.5) / camera.focal_length_x,
        tanfovy=(camera.height * 0.5) / camera.focal_length_y,
        bg=background,
        scale_modifier=1.0,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=gaussians.active_sh_degree,
        campos=camera_center,
        prefiltered=False,
        debug=False,
        antialiasing=False
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_image, radii, depth_image = rasterizer(
        means3D=gaussians.get_xyz,
        means2D=None,
        shs=gaussians.get_features,
        colors_precomp=None,
        opacities=gaussians.get_opacity,
        scales=gaussians.get_scaling,
        rotations=gaussians.get_rotation,
        cov3D_precomp=None,
    )
    rendered_image = rendered_image.clamp(0, 1)

    gt = torch.Tensor(np.moveaxis(image / 255., -1, 0))
    rendered_image = rendered_image.cpu()

    vis = torch.cat([gt, rendered_image], 2)
    torchvision.utils.save_image(vis, out_dir / f"out_{args.frame}.jpg")


if __name__ == "__main__":
    main()
