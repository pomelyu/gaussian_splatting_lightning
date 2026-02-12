import argparse
from pathlib import Path
import time
import cv2
import imageio as iio
import numpy as np
import pycolmap
import torch
import torchvision
from diff_gaussian_rasterization import GaussianRasterizationSettings
from diff_gaussian_rasterization import GaussianRasterizer

from gs_lightning.modules import GaussianModel
from gs_lightning.rasterize import rasterize_gaussian
from gs_lightning.utils.camera import get_projection_matrix



def get_timestamp():
    torch.cuda.synchronize()
    return time.perf_counter()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="path to training checkpoint(.ply)")
    parser.add_argument("--colmap", type=str, help="colmap project used in training", default="data/treehill/sparse/0")
    parser.add_argument("--image", type=str, help="colmap images used in training", default="data/treehill/images")
    parser.add_argument("--down_scale", type=int, help="down scale image", default=1)
    parser.add_argument("--frame", "-n", type=int, default=10, help="frame number")
    parser.add_argument("--output", "-o", type=str, default="results")
    parser.add_argument("--use_pytorch", action="store_true", help="use pytorch insted cuda implementation")
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

    reconstruction = pycolmap.Reconstruction(args.colmap)
    # {1: Camera(camera_id=1, model=PINHOLE, width=5068, height=3326, params=[4219.170711, 4205.602294, 2534.000000, 1663.000000] (fx, fy, cx, cy))}
    cameras = reconstruction.cameras
    # {1: Image(image_id=1, camera_id=1, name="_DSC8874.JPG", triangulated=657/9322)}
    images = reconstruction.images

    fid = args.frame
    cid = images[fid].camera_id
    image_path = images[fid].name

    scale = 1 / args.down_scale
    image = iio.v3.imread(Path(args.image) / image_path)[..., :3]
    H, W = image.shape[:2]
    H, W = int(H*scale), int(W*scale)
    image = cv2.resize(image, (W, H))

    device = "cuda"
    zfar = 100.0
    znear = 0.01
    camera = cameras[cid]
    world_view_transform = np.eye(4)
    world_view_transform[:, :3] = images[fid].cam_from_world().matrix().T
    world_view_transform = torch.Tensor(world_view_transform).to(device)
    camera_center = world_view_transform.inverse()[3, :3].to(device)
    projection_matrix = get_projection_matrix(camera.focal_length_x, camera.focal_length_y, camera.width, camera.height, znear, zfar).T
    projection_matrix = torch.Tensor(projection_matrix).to(device)
    full_proj_transform = world_view_transform @ projection_matrix

    t0 = get_timestamp()
    if args.use_pytorch:
        with torch.no_grad():
            rendered_image, radii, depth_image = rasterize_gaussian(
                means3D=gaussians.get_xyz(),
                opacities=gaussians.get_opacity(),
                scales=gaussians.get_scaling(),
                rotations=gaussians.get_rotation(),
                shs=gaussians.get_features(),
                scale_modifier=1.0,
                image_width=W,
                image_height=H,
                tanfovx=(camera.width * 0.5) / camera.focal_length_x,
                tanfovy=(camera.height * 0.5) / camera.focal_length_y,
                viewmatrix=world_view_transform,
                projmatrix=full_proj_transform,
                campos=camera_center,
                background=background,
                sh_degree=gaussians.active_sh_degree,
            )
    else:
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
    t1 = get_timestamp()
    print(f"rendering {W:d}x{H:d} takes: {t1 - t0:.2f}s = {(t1 - t0)*1000:.2f}ms")

    rendered_image = rendered_image.clamp(0, 1)
    depth_image = depth_image.clamp(0, 1)

    gt = torch.Tensor(np.moveaxis(image / 255., -1, 0))
    rendered_image = rendered_image.cpu()
    depth_image = torch.concat([depth_image, depth_image, depth_image], 0).cpu()

    vis = torch.cat([gt, rendered_image, depth_image], 2)
    out_path = out_dir / f"out_{args.frame}.jpg"
    torchvision.utils.save_image(vis, out_path)
    print(f"Save result to {out_path}")


if __name__ == "__main__":
    main()
