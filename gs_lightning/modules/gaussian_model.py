from typing import List
from typing import Tuple

import kornia as K
import numpy as np
import torch
from plyfile import PlyData
from plyfile import PlyElement
from torch import nn

from gs_lightning.rasterizer_python.forward import computeConv3D
from gs_lightning.utils.colmap import get_nerf_norm
from gs_lightning.utils.math import distCUDA2
from gs_lightning.utils.math import inverse_sigmoid
from gs_lightning.utils.sh import rgb2sh0


class GaussianModel(nn.Module):
    PARAMETER_NAMES = [
        "xyz",
        "features_dc",
        "features_rest",
        "opacity",
        "scaling",
        "rotation",
    ]

    def __init__(
        self,
        sh_degree: int = 3,
        colmap_ply: str = None,
        colmap_path: str = None,
        use_screensize_threshold: bool = True,
    ):
        super().__init__()

        # There is a bug in the official implementation, so this screensize threshold doesn't work
        # In detail, self.max_radii2D has been set to 0 in the previous function
        # (i.e. densification_postfix in both densify_and_clone and densify_and_split),
        # so self.max_radii2D > threshold always return False
        #
        # https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/scene/gaussian_model.py#L462
        # https://github.com/graphdeco-inria/gaussian-splatting/issues/820
        self.use_screensize_threshold = use_screensize_threshold

        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0

        self.activation_opacity = torch.sigmoid
        self.inversed_activation_opacity = inverse_sigmoid

        self.activation_scaling = torch.exp
        self.inversed_activation_scaling = torch.log

        if colmap_ply is not None:
            self.initialize(colmap_ply)

        # In official code, it's called scene.cameras_extent = getNerfppNorm(cam_info)["radius"]
        # It's the scale of the scene
        if colmap_path is not None:
            self.spatial_scale = get_nerf_norm(colmap_path)["radius"]
        else:
            self.spatial_scale = None

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

        self.register_buffer("xyz_grad_accum", torch.zeros(N))
        self.register_buffer("xyz_grad_count", torch.zeros(N))
        self.register_buffer("max_radii2D", torch.zeros(N))

        # TODO: add self._exposure

    # Load & Dump training results
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

    # For Densification
    @torch.no_grad()
    def update_max_radii2D(self, radii: torch.Tensor, visible_mask: torch.Tensor) -> None:
        self.max_radii2D[visible_mask] = torch.max(self.max_radii2D[visible_mask], radii[visible_mask])

    @torch.no_grad()
    def update_xyz_gradient(self, screenspace_gradient: torch.Tensor, visible_mask: torch.Tensor) -> None:
        self.xyz_grad_accum[visible_mask] += torch.norm(screenspace_gradient[visible_mask, :2], dim=1)
        self.xyz_grad_count[visible_mask] += 1

    @torch.no_grad()
    def densify_and_prune(
        self,
        densify_grad_threshold: float,
        clone_size_threshold: float,
        prune_opacity_threshold: float,
        prune_size_threshold: float,
        prune_screensize_threshold: float = None,
    ) -> List[int]:
        preserve_idx = self._prune_gaussian(
            prune_opacity_threshold,
            prune_size_threshold,
            prune_screensize_threshold,
        )

        xyz_grad = self.xyz_grad_accum / self.xyz_grad_count
        xyz_grad[xyz_grad.isnan()] = 0.0

        bad_mask = (xyz_grad >= densify_grad_threshold)
        gaussian_size = torch.max(self.get_scaling(), dim=1)[0]
        clone_size_threshold = clone_size_threshold * self.spatial_scale
        bad_small_idx = torch.logical_and(bad_mask, gaussian_size < clone_size_threshold).nonzero().squeeze(-1)
        bad_large_idx = torch.logical_and(bad_mask, gaussian_size >= clone_size_threshold).nonzero().squeeze(-1)

        self._clone_gaussian(bad_small_idx)
        self._split_gaussian(bad_large_idx)

        return preserve_idx

    @torch.no_grad()
    def _prune_gaussian(
        self,
        opacity_threshold: float,
        prune_size_threshold: float,
        prune_screensize_threshold: float = None
    ) -> List[int]:
        preserve_mask = (self.get_opacity() > opacity_threshold).squeeze(-1)
        if prune_screensize_threshold is not None:
            if self.use_screensize_threshold:
                preserve_mask = torch.logical_and(preserve_mask, self.max_radii2D < prune_screensize_threshold)
            gaussian_size = torch.max(self.get_scaling(), dim=1)[0]
            preserve_mask = torch.logical_and(preserve_mask, gaussian_size < prune_size_threshold * self.spatial_scale)

        self._xyz = nn.Parameter(self._xyz[preserve_mask])
        self._features_dc = nn.Parameter(self._features_dc[preserve_mask])
        self._features_rest = nn.Parameter(self._features_rest[preserve_mask])
        self._opacity = nn.Parameter(self._opacity[preserve_mask])
        self._scaling = nn.Parameter(self._scaling[preserve_mask])
        self._rotation = nn.Parameter(self._rotation[preserve_mask])

        self.max_radii2D = self.max_radii2D[preserve_mask]
        self.xyz_grad_accum = self.xyz_grad_accum[preserve_mask]
        self.xyz_grad_count = self.xyz_grad_count[preserve_mask]

        return preserve_mask.nonzero().squeeze(-1)

    @torch.no_grad()
    def _clone_gaussian(self, selection_idx: torch.Tensor):
        # clone a small gaussian to better cover the under-reconstructed region
        self._add_gaussian(
            xyz=self._xyz[selection_idx].clone(),
            features_dc=self._features_dc[selection_idx].clone(),
            features_rest=self._features_rest[selection_idx].clone(),
            opacity=self._opacity[selection_idx].clone(),
            scaling=self._scaling[selection_idx].clone(),
            rotation=self._rotation[selection_idx].clone(),
        )

    @torch.no_grad()
    def _split_gaussian(self, selection_idx: torch.Tensor):
        # split a large gaussian to better fit the under-reconstructed region
        displace_std = self.get_scaling()[selection_idx]
        displace_mean = torch.zeros_like(self.get_xyz()[selection_idx])
        displace = torch.normal(mean=displace_mean, std=displace_std)
        R = K.geometry.Quaternion(self.get_rotation()[selection_idx]).matrix()

        xyz = self.get_xyz()[selection_idx] + torch.bmm(R, displace.unsqueeze(-1)).squeeze(-1)
        self._xyz[selection_idx] = xyz[:]

        scaling = self.inversed_activation_scaling(self.get_scaling()[selection_idx] / 1.6)
        self._scaling[selection_idx] = scaling[:]

        self._clone_gaussian(selection_idx)

    @torch.no_grad()
    def _add_gaussian(
        self,
        xyz: torch.Tensor,
        features_dc: torch.Tensor,
        features_rest: torch.Tensor,
        opacity: torch.Tensor,
        scaling: torch.Tensor,
        rotation: torch.Tensor,
    ):
        N = len(xyz)
        self._xyz = nn.Parameter(torch.cat([self._xyz, xyz], dim=0))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc, features_dc], dim=0))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest, features_rest], dim=0))
        self._opacity = nn.Parameter(torch.cat([self._opacity, opacity], dim=0))
        self._scaling = nn.Parameter(torch.cat([self._scaling, scaling], dim=0))
        self._rotation = nn.Parameter(torch.cat([self._rotation, rotation], dim=0))

        self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros(N).to(self.max_radii2D)])
        self.xyz_grad_accum = torch.cat([self.xyz_grad_accum, torch.zeros(N).to(self.xyz_grad_accum)])
        self.xyz_grad_count = torch.cat([self.xyz_grad_count, torch.zeros(N).to(self.xyz_grad_count)])

    @torch.no_grad()
    def reset_opacity(self):
        new_opacity = torch.min(self.get_opacity(), torch.ones_like(self._opacity) * 0.01)
        new_opacity = self.inversed_activation_opacity(new_opacity)
        self._opacity[:] = new_opacity[:]

    def reset_max_radii2D(self):
        self.max_radii2D.fill_(0.0)

    def reset_xyz_gradient(self):
        self.xyz_grad_accum.fill_(0.0)
        self.xyz_grad_count.fill_(0.0)

    def step_sh_degree(self):
        self.active_sh_degree = min(self.active_sh_degree + 1, self.max_sh_degree)

    def ready_for_training(self) -> bool:
        if not hasattr(self, "_xyz"):
            raise RuntimeError("colmap_ply is required for training")
        if self.spatial_scale is None:
            raise RuntimeError("colmap_path is required for training")
        return True

    def ready_for_inference(self) -> bool:
        if not hasattr(self, "_xyz"):
            raise RuntimeError("load_model_ply should be executred before inference")
        return True

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

    def get_covariance(self, scaling_modifier: float = 1) -> torch.Tensor:
        return computeConv3D(self.get_scaling(), scaling_modifier, self.get_rotation())
