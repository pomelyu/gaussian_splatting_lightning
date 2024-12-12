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

from gs_lightning.modules import GaussianModel
from gs_lightning.utils.camera import get_projection_matrix


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
    gaussians.load_model_ply(args.model)
    gaussians.ready_for_inference()
    gaussians.to(device)

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
    H, W = int(H*scale), int(W*scale)
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
    projection_matrix = torch.Tensor(projection_matrix).to(device)
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
        means3D=gaussians.get_xyz(),
        means2D=None,
        shs=gaussians.get_features(),
        colors_precomp=None,
        opacities=gaussians.get_opacity(),
        scales=gaussians.get_scaling(),
        rotations=gaussians.get_rotation(),
        cov3D_precomp=None,
    )

    rendered_image = rendered_image.clamp(0, 1)

    gt = torch.Tensor(np.moveaxis(image / 255., -1, 0))
    rendered_image = rendered_image.cpu()

    vis = torch.cat([gt, rendered_image], 2)
    out_path = out_dir / f"out_{args.frame}.jpg"
    torchvision.utils.save_image(vis, out_path)
    print(f"Save result to {out_path}")


if __name__ == "__main__":
    main()
