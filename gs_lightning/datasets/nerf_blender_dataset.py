import json
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

import cv2
import mlconfig
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import trange

from gs_lightning.utils.camera import get_projection_matrix
from gs_lightning.utils.image import load_image

from .dataloader import ConfigTrainDataloader
from .dataloader import ConfigValidDataloader


def calculate_camera_focal(image_width: int, camera_angle_x: float) -> float:
    return 0.5 * image_width / np.tan(0.5 * camera_angle_x)


@mlconfig.register()
class NerfBlenderDataset(Dataset):
    def __init__(
        self,
        dataroot: str,
        split: str = "train",
        resize_to: Optional[int] = None,
        downscale: Optional[float] = None,
        white_background: bool = True,
        z_near: float = 0.01,
        z_far: float = 100.0,
        preload_data: bool = True,
    ):
        """
        Dataset for NeRF Blender Synthetic scenes (nerf_synthetic).

        Args:
            dataroot: path to the blender scene folder (e.g. .../nerf_synthetic/lego)
            split: one of "train", "val", "test"
            resize_to: resize the longest side of the image to this value
            downscale: downscale factor applied to H and W
            white_background: whether to composite RGBA images onto a white background
            z_near: near clipping plane
            z_far: far clipping plane
            preload_data: if True, load all images into RAM during init
        """
        super().__init__()

        self.dataroot = Path(dataroot)
        self.resize_to = resize_to
        self.downscale = downscale
        self.z_near = z_near
        self.z_far = z_far
        self.background = torch.Tensor([1, 1, 1]) if white_background else torch.Tensor([0, 0, 0])

        # Load transforms JSON
        meta_path = self.dataroot / f"transforms_{split}.json"
        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.camera_angle_x = meta["camera_angle_x"]
        self.frames = meta["frames"]

        self.cached_data = {}
        if preload_data:
            for index in trange(len(self.frames), desc=f"Preload {split} data"):
                self.cached_data[index] = self.build_item(index)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        if index in self.cached_data:
            return self.cached_data[index]

        data = self.build_item(index)
        self.cached_data[index] = data
        return data

    def build_item(self, index: int) -> dict:
        frame = self.frames[index]
        image_path = self.dataroot / f"{frame['file_path']}.png"

        # Load RGBA image and composite onto background
        image_rgba = load_image(str(image_path))  # (H, W, 4) uint8
        if image_rgba.shape[-1] == 4:
            image_rgb = image_rgba[..., :3].astype(np.float32) / 255.0
            alpha = image_rgba[..., 3:].astype(np.float32) / 255.0
            bg_np = self.background.numpy()
            image_rgb = image_rgb * alpha + (1.0 - alpha) * bg_np[None, None, :]
        elif image_rgba.shape[-1] == 3:
            image_rgb = image_rgba
        else:
            raise RuntimeError(f"Unexpected image format: {image.shape}")

        # Resize
        H, W = image_rgb.shape[:2]
        if self.downscale is not None:
            H, W = int(H * self.downscale), int(W * self.downscale)
        elif self.resize_to is not None:
            scale = self.resize_to / max(H, W)
            H, W = int(H * scale), int(W * scale)

        if (H, W) != image_rgb.shape[:2]:
            image_rgb = cv2.resize(image_rgb, (W, H), interpolation=cv2.INTER_LINEAR)

        # (C, H, W)
        image = torch.Tensor(np.moveaxis(image_rgb, -1, 0))

        # Camera intrinsics
        camera_angle_x = frame.get("camera_angle_x", self.camera_angle_x)
        focal = calculate_camera_focal(W, camera_angle_x)

        # Camera extrinsics: blender c2w -> world_view_transform (w2c)
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)  # (4, 4)
        # Convert from blender coordinate (right, up, -forward) to OpenGL/colmap (right, down, forward)
        # Flip Y and Z axes
        c2w[:3, 1:3] *= -1

        w2c = np.linalg.inv(c2w)
        # world_view_transform is stored as row-major (transposed) to match the ColmapDataset convention
        world_view_transform = torch.Tensor(w2c.T)
        camera_center = torch.Tensor(c2w[:3, 3])

        # Projection matrix
        projection_matrix = get_projection_matrix(
            fx=focal,
            fy=focal,  # blender uses square pixels
            w=W,
            h=H,
            znear=self.z_near,
            zfar=self.z_far,
        ).T
        projection_matrix = torch.Tensor(projection_matrix)
        full_proj_transform = world_view_transform @ projection_matrix

        data = dict(
            image=image,
            tanfovx=(W * 0.5) / focal,
            tanfovy=(H * 0.5) / focal,
            background=self.background,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            campos=camera_center,
        )
        return data


@mlconfig.register()
class NerfBlenderDataModule(LightningDataModule):
    def __init__(
        self,
        num_iters: int,
        dataroot: str,
        train_split: str = "train",
        valid_split: str = "test",
        resize_to: Optional[int] = None,
        downscale: Optional[float] = None,
        white_background: bool = True,
        z_near: float = 0.01,
        z_far: float = 100.0,
        preload_data: bool = True,
    ):
        super().__init__()

        def create_dataset(split):
            return NerfBlenderDataset(
                dataroot=dataroot,
                split=split,
                resize_to=resize_to,
                downscale=downscale,
                white_background=white_background,
                z_near=z_near,
                z_far=z_far,
                preload_data=preload_data,
            )

        self.train_dataset = create_dataset(train_split)
        self.valid_dataset = create_dataset(valid_split)
        self.num_iters = num_iters

    def train_dataloader(self) -> DataLoader:
        return ConfigTrainDataloader(
            dataset=self.train_dataset,
            num_iters=self.num_iters,
            batch_size=1,
            num_workers=0,
        )

    def val_dataloader(self) -> DataLoader:
        return ConfigValidDataloader(
            dataset=self.valid_dataset,
            batch_size=1,
            num_workers=0,
        )
