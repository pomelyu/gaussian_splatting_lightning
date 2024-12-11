from typing import List
from typing import Tuple

import kornia as K
import numpy as np
import torch
from plyfile import PlyData
from plyfile import PlyElement
from torch import nn

from gs_lightning.utils.math import distCUDA2
from gs_lightning.utils.math import inverse_sigmoid
from gs_lightning.utils.sh import rgb2sh0


class GaussianModel(nn.Module):
    def __init__(
        self,
        sh_degree: int = 3,
        colmap_ply: str = None,
    ):
        super().__init__()

        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0

        self.activation_opacity = torch.sigmoid
        self.inversed_activation_opacity = inverse_sigmoid

        self.activation_scaling = torch.exp
        self.inversed_activation_scaling = torch.log

        if colmap_ply is not None:
            self.initialize(colmap_ply)

    def initialize(self, colmap_ply: str) -> None:
        plyData = PlyData.read(colmap_ply)
        vertices: PlyElement = plyData["vertex"]
        xyz = torch.Tensor(np.stack([vertices["x"], vertices["y"], vertices["z"]], -1))
        color = torch.Tensor(np.stack([vertices["red"], vertices["green"], vertices["blue"]], -1) / 255.)
        # We don't use normal information in this project
        # normal = torch.stack([vertices["nx"], vertices["ny"], vertices["nz"]], -1)

        N = len(xyz)
        # Note that n-order spherical-harmonic(sh) contains (n+1)^2 parameters
        # We use different learning rate for base band and high order bands to avoid strong view-dependent overfitting
        # https://github.com/graphdeco-inria/gaussian-splatting/issues/438#issuecomment-1798848156
        sh_params = torch.zeros(N, 3, (self.max_sh_degree + 1)**2)
        sh_params[:, :, 0] = rgb2sh0(color)

        # use quaternion for rotation
        rotation = torch.zeros((N, 4))
        rotation[:, 0] = 1

        # Find the average distance from the N(4?) closest points.
        # Then use it as a initial scale/size of a gaussian splat.
        # 
        # The function is only used for initialization
        # Unlike original code that uses distCUDA2 from simple-knn,
        # I use a python implementation here
        dist = torch.clamp_min(distCUDA2(xyz), 0.0000001)
        scale = self.inversed_activation_scaling(torch.sqrt(dist))[..., None].repeat(1, 3)

        # initialize opacity to inverse_sigmoid(0.1),
        # since we'll apply sigmoid to make sure opacity > 0
        opacity = 0.1 * torch.ones(N, 1)
        opacity = self.inversed_activation_opacity(opacity)

        self._xyz = nn.Parameter(xyz)
        self._features_dc = nn.Parameter(sh_params[..., :1].transpose(1, 2).contiguous())
        self._features_rest = nn.Parameter(sh_params[..., 1:].transpose(1, 2).contiguous())
        self._scaling = nn.Parameter(scale)
        self._rotation = nn.Parameter(rotation)
        self._opacity = nn.Parameter(opacity)

        # TODO: add self._exposure

    def load_model_ply(self, ply_path: str) -> None:
        plyData = PlyData.read(ply_path)
        vertices: PlyElement = plyData["vertex"]
        xyz = torch.Tensor(np.stack([vertices["x"], vertices["y"], vertices["z"]], -1))
        N = len(xyz)
        assert xyz.shape == (N, 3), xyz.shape

        features_dc = GaussianModel.load_vertices_properties(vertices, "f_dc").reshape(N, 1, 3)
        features_rest = GaussianModel.load_vertices_properties(vertices, "f_rest").reshape(N, -1, 3)
        scaling = GaussianModel.load_vertices_properties(vertices, "scale").reshape(N, 3)
        rotation = GaussianModel.load_vertices_properties(vertices, "rot").reshape(N, 4)
        opacity = GaussianModel.load_vertices_properties(vertices, "opacity").reshape(N, 1)

        self._xyz = nn.Parameter(xyz)
        self._features_dc = nn.Parameter(features_dc)
        self._features_rest = nn.Parameter(features_rest)
        self._scaling = nn.Parameter(scaling)
        self._rotation = nn.Parameter(rotation)
        self._opacity = nn.Parameter(opacity)

        self.active_sh_degree = int(np.sqrt(features_rest.shape[-1] + 1))

    def step_sh_degree(self):
        self.active_sh_degree = min(self.active_sh_degree + 1, self.max_sh_degree)

    @classmethod
    def load_vertices_properties(cls, vertices: PlyElement, name: str) -> torch.Tensor:
        property_names = [p.name for p in vertices.properties if p.name.startswith(name)]
        property_names = list(sorted(property_names))
        properties = [vertices[name] for name in property_names]
        properties = np.stack(properties, -1)
        return torch.Tensor(properties)

    @classmethod
    def create_structure_dtype(cls, data: torch.Tensor, prefix: str) -> Tuple[np.ndarray, List[str]]:
        data = data.flatten(1).detach().cpu().numpy()
        structure_dtype = []
        for i in range(data.shape[1]):
            structure_dtype.append((f"{prefix}_{i}", "f4"))
        return data, structure_dtype

    def save_ply(self, ply_path: str) -> bool:
        xyz = self._xyz.detach().cpu().numpy()
        xyz_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]

        normal = np.zeros_like(xyz)
        normal_dtype = [("nx", "f4"), ("ny", "f4"), ("nz", "f4")]

        f_dc, f_dc_dtype = GaussianModel.create_structure_dtype(self._features_dc.transpose(1, 2), "f_dc")
        f_rest, f_rest_dtype = GaussianModel.create_structure_dtype(self._features_rest.transpose(1, 2), "f_rest")

        opacity = self._opacity.detach().cpu().numpy()
        opacity_dtype = [("opacity", "f4")]

        scale, scale_dtype = GaussianModel.create_structure_dtype(self._scaling, "scale")
        rotation, rotation_dtype = GaussianModel.create_structure_dtype(self._rotation, "rot")

        attributes = np.concatenate([xyz, normal, f_dc, f_rest, opacity, scale, rotation], -1)
        structure_dtype = xyz_dtype + normal_dtype + f_dc_dtype + f_rest_dtype + opacity_dtype + scale_dtype + rotation_dtype

        attributes = np.array([tuple(attr) for attr in attributes], dtype=structure_dtype)
        element = PlyElement.describe(attributes, "vertex")
        PlyData([element]).write(ply_path)

    def get_xyz(self) -> torch.Tensor:
        return self._xyz
    
    def get_features(self) -> torch.Tensor:
        return torch.cat([self._features_dc, self._features_rest], 1)

    def get_opacity(self) -> torch.Tensor:
        return self.activation_opacity(self._opacity)
    
    def get_scaling(self) -> torch.Tensor:
        return self.activation_scaling(self._scaling)
    
    def get_rotation(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self._rotation)

    # def get_covariance(self, scaling_modifier: float = 1) -> torch.Tensor:
    #     # calculate covariance matrix from scaling and rotation
    #     # eq6. Covariance = R @ S @ S^T @ R^T
    #     R = K.geometry.Quaternion(self.get_rotation()).matrix()      # (N, 3, 3)
    #     S = torch.diag_embed(self.get_scaling() * scaling_modifier)  # (N, 3, 3)
    #     L = R @ S
    #     covar = L @ L.transpose(1, 2)                           # (N, 3, 3)
    #     # use the low diagnoal elements to make sure the matrix is symmetric
    #     out = torch.stack([
    #         covar[:, 0, 0], covar[:, 0, 1], covar[:, 0, 2],
    #         covar[:, 1, 1], covar[:, 1, 2], covar[:, 2, 2],
    #     ], -1)
    #     return out
