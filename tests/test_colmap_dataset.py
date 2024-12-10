import cv2
import numpy as np
import torch
from pycolmap import MapPoint3DIdToPoint3D
from torch.utils.data import DataLoader
from tqdm import tqdm

from gs_lightning.datasets import ColmapDataset
from gs_lightning.utils.os import mkdir

COLMAP_PATH = "C:/Users/cychien-desktop/Documents/database/NeRF/Kanade_360_colmap/sparse/0"
IMAGE_FOLDER = "C:/Users/cychien-desktop/Documents/database/NeRF/Kanade_360_colmap/images"

def test_colmap_dataset():
    dataset = ColmapDataset(
        colmap_path=COLMAP_PATH,
        image_folder=IMAGE_FOLDER,
        resize_to=960,
    )
    dataloader = DataLoader(dataset, batch_size=4, drop_last=False, shuffle=False)

    # {1: Point3D(xyz=[0.116297, -0.52869, -0.570282], color=[80, 67, 48], error=0.442101, track=Track(length=2))}
    points: MapPoint3DIdToPoint3D = dataset.reconstruction.points3D
    points3d = torch.Tensor(np.array([[*pt.xyz, 1.0] for pt in points.values()]))
    points3d_colors = np.array([pt.color for pt in points.values()], dtype=np.uint8)

    out_dir = mkdir("results/test_colmap_dataset")
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        vis = []
        for image, projmatrix in zip(data["image"], data["projmatrix"]):
            image = np.moveaxis(image.numpy() * 255, 0, -1).astype(np.uint8)
            H, W = image.shape[:2]
            # project points to 2d
            # Note that the projection matrix project image to x, y: [-1, 1] and z: [0, 1]
            points2d = points3d @ projmatrix
            points2d[:, 0] = ((points2d[:, 0] / points2d[:, 2]) * 0.5 + 0.5) * W
            points2d[:, 1] = ((points2d[:, 1] / points2d[:, 2]) * 0.5 + 0.5) * H
            points2d = points2d[:, :2]
            # draw projected 2d points
            canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
            for pt, color in zip(points2d.int().numpy(), points3d_colors):
                # RGB -> BGR
                color = tuple(color.tolist()[::-1])
                cv2.circle(canvas, pt, 1, color, -1)
            vis.append(np.concatenate([image, canvas], 0))
        vis = np.concatenate(vis, 1)
        cv2.imwrite(str(out_dir / f"image_{i:0>3d}.jpg"), cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
    test_colmap_dataset()
