import numpy as np
import pycolmap


def get_nerf_norm(colmap_path: str):
    reconstruction = pycolmap.Reconstruction(colmap_path)
    # {1: Image(image_id=1, camera_id=1, name="_DSC8874.JPG", triangulated=657/9322)}
    image_info = reconstruction.images

    cameras = []
    for info in image_info.values():
        w2c = np.eye(4)
        w2c[:3, :] = info.cam_from_world().matrix()
        c2w = np.linalg.inv(w2c)
        cameras.append(c2w[:3, -1:])

    cameras = np.stack(cameras)     # (N, 3)
    center = np.mean(cameras, axis=0, keepdims=True)    # (N, 1)
    max_dist = np.linalg.norm(cameras - center, axis=1).max()

    radius = max_dist * 1.1
    translate = -center
    return {"translate": translate, "radius": radius}
