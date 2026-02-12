import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from gs_lightning.datasets import NerfBlenderDataset
from gs_lightning.utils.os import mkdir

DATAPATH = "data/nerf_blender/lego"

def test_nerf_blender_dataset():
    dataset = NerfBlenderDataset(
        dataroot=DATAPATH,
        resize_to=960
    )
    dataloader = DataLoader(dataset, batch_size=4, drop_last=False, shuffle=False)

    out_dir = mkdir("results/test_nerf_blender_dataset")
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        vis = []
        for image, projmatrix in zip(data["image"], data["projmatrix"]):
            image = np.moveaxis(image.numpy() * 255, 0, -1).astype(np.uint8)
            vis.append(image)
        vis = np.concatenate(vis, 1)
        cv2.imwrite(str(out_dir / f"image_{i:0>3d}.jpg"), cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
    test_nerf_blender_dataset()
