from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

import cv2
import mlconfig
import numpy as np
import pycolmap
import torch
from pycolmap import MapCameraIdToCamera
from pycolmap import MapImageIdToImage
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import trange

from gs_lightning.utils.camera import get_projection_matrix
from gs_lightning.utils.image import load_image

from .dataloader import ConfigTrainDataloader
from .dataloader import ConfigValidDataloader


@mlconfig.register()
class ColmapDataset(Dataset):
    def __init__(
        self,
        colmap_path: str,
        image_folder: str,
        image_idx: Optional[Union[List[int], str]] = None,
        mask_folder: Optional[str] = None,
        resize_to: Optional[int] = None,
        downscale: Optional[float] = None,
        white_background: bool = False,
        z_near: float = 0.01,
        z_far: float = 100.0,
        preload_data: bool = True,
    ):
        super().__init__()

        self.resize_to = resize_to
        self.downscale = downscale
        self.z_near = z_near
        self.z_far = z_far
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.background = torch.Tensor([1, 1, 1]) if white_background else torch.Tensor([0, 0, 0])

        reconstruction = pycolmap.Reconstruction(colmap_path)
        # {1: Camera(camera_id=1, model=PINHOLE, width=5068, height=3326, params=[4219.170711, 4205.602294, 2534.000000, 1663.000000] (fx, fy, cx, cy))}
        camera_info: MapCameraIdToCamera = reconstruction.cameras
        # {1: Image(image_id=1, camera_id=1, name="_DSC8874.JPG", triangulated=657/9322)}
        image_info: MapImageIdToImage = reconstruction.images

        self.image_indices = ColmapDataset.load_image_idx(image_idx)
        if self.image_indices is None:
            self.image_indices = list(image_info.keys())

        self.cached_data = {}
        if preload_data:
            for index in trange(self.__len__(), desc="Preload data"):
                self.cached_data[index] = self.build_item(index, camera_info, image_info)
        else:
            self.camera_info = camera_info
            self.image_info = image_info

    def __len__(self):
        return len(self.image_indices)
    
    def __getitem__(self, index):
        if index in self.cached_data:
            return self.cached_data[index]

        data = self.build_item(index, self.camera_info, self.image_info)
        self.cached_data[index] = data
        return data
    
    def build_item(self, index, camera_info: MapCameraIdToCamera, image_info: MapImageIdToImage) -> dict:
        image_idx = self.image_indices[index]
        image_info: MapImageIdToImage = image_info[image_idx]
        camera_info: MapCameraIdToCamera = camera_info[image_info.camera_id]

        image_name = image_info.name
        image = self.load_image_to_tensor(self.image_folder, image_name)

        world_view_transform = np.eye(4)
        world_view_transform[:, :3] = image_info.cam_from_world.matrix().T
        world_view_transform = torch.Tensor(world_view_transform)
        camera_center = world_view_transform.inverse()[3, :3]
        projection_matrix = get_projection_matrix(
            camera_info.focal_length_x,
            camera_info.focal_length_y,
            camera_info.width,
            camera_info.height,
            self.z_near,
            self.z_far,
        ).T
        projection_matrix = torch.Tensor(projection_matrix)
        full_proj_transform = world_view_transform @ projection_matrix

        _, H, W = image.shape
        camera_intrinsic = torch.eye(3)
        camera_intrinsic[0, 0] = camera_info.focal_length_x * (W /camera_info.width)
        camera_intrinsic[1, 1] = camera_info.focal_length_y * (H /camera_info.height)
        camera_intrinsic[0, 2] = W * 0.5
        camera_intrinsic[1, 2] = H * 0.5

        data = dict(
            image=image,
            tanfovx=(camera_info.width * 0.5) / camera_info.focal_length_x,
            tanfovy=(camera_info.height * 0.5) / camera_info.focal_length_y,
            background=self.background,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            campos=camera_center,
            camera_intrinsic=camera_intrinsic,
        )
        return data

    @classmethod
    def load_image_idx(cls, image_idx: Optional[Union[List[int], str]] = None) -> Optional[List[int]]:
        if image_idx is None:
            return None
        elif isinstance(image_idx, list):
            image_idx = np.array(image_idx)
        else:
            image_idx = np.loadtxt(image_idx, delimiter=",", dtype=np.int64)

        if (image_idx.ndim) != 1:
            raise ValueError("image_idx should be a list of integar")
        return image_idx

    def load_image_to_tensor(self, image_folder: str, image_name: str) -> torch.Tensor:
        image = load_image(Path(image_folder) / image_name)
        H, W = image.shape[:2]
        if self.downscale is not None:
            H, W = int(H*self.downscale), int(W*self.downscale)
        elif self.resize_to is not None:
            scale = self.resize_to / max(H, W)
            H, W = int(H*scale), int(W*scale)
        else:
            raise ValueError("Either 'downscale' or 'resize_to' must be set")

        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
        image = image / 255.

        if self.mask_folder is not None:
            mask = load_image(Path(self.mask_folder) / image_name)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
            mask = mask / 255.
            image = image * mask

        image = torch.Tensor(np.moveaxis(image, -1, 0))
        return image


@mlconfig.register()
class ColmapDataModule(LightningDataModule):
    def __init__(
        self,
        num_iters: int,
        colmap_path: str,
        image_folder: str,
        train_idx_file: str,
        valid_idx_file: str,
        mask_folder: Optional[str] = None,
        resize_to: Optional[int] = None,
        downscale: Optional[float] = None,
        white_background: bool = False,
        z_near: float = 0.01,
        z_far: float = 100.0,
        preload_data: bool = True,
    ): 
        super().__init__()

        def create_dataset(idx_file):
            return ColmapDataset(
                colmap_path=colmap_path,
                image_folder=image_folder,
                image_idx=idx_file,
                mask_folder=mask_folder,
                resize_to=resize_to,
                downscale=downscale,
                white_background=white_background,
                z_near=z_near,
                z_far=z_far,
                preload_data=preload_data,
            )

        self.train_dataset = create_dataset(train_idx_file)
        self.valid_dataset = create_dataset(valid_idx_file)
        self.num_iters = num_iters

    def train_dataloader(self) -> DataLoader:
        return ConfigTrainDataloader(
            dataset=self.train_dataset,
            num_iters=self.num_iters,
            batch_size=1,
            num_workers=0,
        )
    
    def val_dataloader(self):
        return ConfigValidDataloader(
            dataset=self.valid_dataset,
            batch_size=1,
            num_workers=0,
        )
