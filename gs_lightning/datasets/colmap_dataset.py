from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

import cv2
import numpy as np
import pycolmap
import torch
from pycolmap import MapCameraIdToCamera
from pycolmap import MapImageIdToImage
from torch.utils.data import Dataset

from gs_lightning.utils import get_projection_matrix
from gs_lightning.utils import load_image


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
    ):
        super().__init__()

        self.resize_to = resize_to
        self.downscale = downscale
        self.z_near = z_near
        self.z_far = z_far
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.background = torch.Tensor([1, 1, 1]) if white_background else torch.Tensor([0, 0, 0])

        self.reconstruction = pycolmap.Reconstruction(colmap_path)
        # {1: Camera(camera_id=1, model=PINHOLE, width=5068, height=3326, params=[4219.170711, 4205.602294, 2534.000000, 1663.000000] (fx, fy, cx, cy))}
        self.camera_info: MapCameraIdToCamera = self.reconstruction.cameras
        # {1: Image(image_id=1, camera_id=1, name="_DSC8874.JPG", triangulated=657/9322)}
        self.image_info: MapImageIdToImage = self.reconstruction.images

        self.image_idxes = ColmapDataset.load_image_idx(image_idx)
        if self.image_idxes is None:
            self.image_idxes = list(self.image_info.keys())
        self.cached_data = {}

    def __len__(self):
        return len(self.image_idxes)
    
    def __getitem__(self, index):
        if index in self.cached_data:
            return self.cached_data[index]

        image_idx = self.image_idxes[index]
        image_info: MapImageIdToImage = self.image_info[image_idx]
        camera_info: MapCameraIdToCamera = self.camera_info[image_info.camera_id]

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

        data = dict(
            image=image,
            tanfovx=(camera_info.width * 0.5) / camera_info.focal_length_x,
            tanfovy=(camera_info.height * 0.5) / camera_info.focal_length_y,
            background=self.background,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            campos=camera_center,
        )
        self.cached_data[index] = data
        return data

    @classmethod
    def load_image_idx(cls, image_idx: Union[List[int], str]) -> Optional[List[int]]:
        if image_idx is None:
            return None
        elif isinstance(image_idx, list):
            image_idx = np.array(image_idx)
        else:
            image_idx = np.loadtxt(image_idx, delimiter=",")

        if (image_idx.shape) != 1:
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
