from dataclasses import asdict
from dataclasses import dataclass
from typing import Tuple, List

import mlconfig
import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings
from diff_gaussian_rasterization import GaussianRasterizer
from fused_ssim import fused_ssim
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import nn
import torch.utils

from gs_lightning.modules import GaussianModel
from gs_lightning.scheduler import GSWarmUpExponentialDecayScheduler
from gs_lightning.utils.lightning import MLFlowLogger


@dataclass
class CFGTrainer:
    # common parameters
    num_iters: int = 30_000
    print_interval: int = 100
    display_interval: int = 1000
    valid_interval: int = 1000
    # gs parameters
    w_ssim: float = 0.2
    percent_dense: float = 0.01
    opacity_reset_interval: int = 3000
    densify_interval: int = 100
    densify_since: int = 500
    densify_util: int = 15_000
    densify_grad_threshold: float = 0.0002
    clone_size_threshold: float = 0.01          # =percent_dense. threshold between clone or split a gaussian
    prune_opacity_threshold: float = 0.005
    prune_size_threshold: float = 0.1
    prune_screensize_threshold: float = 40.0
    sh_degree_step_interval: int = 1000
    # TODO: add depth regularization
    # rendering options
    compute_cov3D_python: bool = False      # compute 3D covariances by gaussian model instead of rasterizer
    convert_SHs_python: bool = False        # compute color value by gaussian model instead of rasterizer
    seperate_SHs: bool = False              # seperate base SH band and high-order SH bands

@dataclass
class CFGModel:
    sh_degree: int = 3
    colmap_ply: str = None
    colmap_path: str = None
    use_screensize_threshold: bool = True

@dataclass
class CFGOptimizer:
    optimizer: DictConfig
    xyz_lr_init: float = 0.00016
    feature_lr: float = 0.0025
    opacity_lr: float = 0.025
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    r_dc2rest: float = 20.0

@dataclass
class CFGScheduler:
    param: str = "xyz"
    lr_init: float = 0.00016
    lr_final: float = 0.000016
    lr_delay_multi: float = 0.001
    lr_delay_step: int = 0
    max_steps: int = 30_000

@mlconfig.register()
class GSLightningModule(LightningModule):
    def __init__(
        self,
        meta: dict,
        cfg_trainer: DictConfig,
        cfg_model: DictConfig,
        cfg_optimizer: DictConfig,
        cfg_scheduler: DictConfig,
        cfg_train_dataloader: DictConfig = None,
        cfg_valid_dataloader: DictConfig = None,
    ):
        super().__init__()

        self.automatic_optimization = False

        self.cfg_trainer: CFGTrainer = CFGTrainer(**cfg_trainer)
        self.cfg_model: CFGModel = CFGModel(**cfg_model)
        self.cfg_optimizer: CFGOptimizer = CFGOptimizer(**cfg_optimizer)
        self.cfg_scheduler: CFGScheduler = CFGScheduler(**cfg_scheduler)
        self.cfg_train_dataloader = cfg_train_dataloader
        self.cfg_valid_dataloader = cfg_valid_dataloader

        self.gaussians = GaussianModel(**asdict(self.cfg_model))
        self.gaussians.ready_for_training()

        self.criterion_recon = nn.L1Loss()
        self.criterion_ssim = lambda x, y: 1 - fused_ssim(x, y)

        self.meta = meta
        self.show_valid_idx = []
        self.cached_results = None

    def train_dataloader(self):
        return mlconfig.instantiate(self.cfg_train_dataloader)

    def val_dataloader(self):
        if self.cfg_valid_dataloader is not None:
            return mlconfig.instantiate(self.cfg_valid_dataloader)
        return

    def configure_optimizers(self):
        # TODO: enable SpareAdam
        spatial_lr_scale = self.gaussians.spatial_scale
        self.cfg_optimizer.xyz_lr_init *= spatial_lr_scale

        lr_features_rest = self.cfg_optimizer.feature_lr / self.cfg_optimizer.r_dc2rest
        parameters = [
            {"params": [self.gaussians._xyz], "lr": self.cfg_optimizer.xyz_lr_init, "name": "xyz"},
            {"params": [self.gaussians._features_dc], "lr": self.cfg_optimizer.feature_lr, "name": "features_dc"},
            {"params": [self.gaussians._features_rest], "lr": lr_features_rest, "name": "features_rest"},
            {"params": [self.gaussians._opacity], "lr": self.cfg_optimizer.opacity_lr, "name": "opacity"},
            {"params": [self.gaussians._scaling], "lr": self.cfg_optimizer.scaling_lr, "name": "scaling"},
            {"params": [self.gaussians._rotation], "lr": self.cfg_optimizer.rotation_lr, "name": "rotation"},
        ]
        optimizer = mlconfig.instantiate(self.cfg_optimizer.optimizer, parameters)

        assert self.cfg_scheduler.param == "xyz"
        self.cfg_scheduler.lr_init *= spatial_lr_scale
        self.cfg_scheduler.lr_final *= spatial_lr_scale
        scheduler = GSWarmUpExponentialDecayScheduler(optimizer=optimizer, **asdict(self.cfg_scheduler))
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def process_data(self, batch: dict) -> dict:
        data_dict = {}
        for k, v in batch.items():
            if len(v) != 1:
                raise RuntimeError("Batch size should be 1")
            data_dict[k] = v[0]
        return data_dict

    def on_train_batch_start(self, batch, batch_idx):
        # Peform densification based on the results of the previous batch
        # It's not possible to do this in the end of a training_step,
        # because this process modifies parameters and thus causes the incorect visuals and validation losses
        if self.cached_results is None:
            return super().on_train_batch_start(batch, batch_idx)
 
        if self.global_step < self.cfg_trainer.densify_util:
            self.densify_gaussian(self.cached_results["radii2D"], self.cached_results["screenspace_gradient"])

        if self.global_step % self.cfg_trainer.opacity_reset_interval == 0:
            self.gaussians.reset_opacity()

        # increate sh_degree
        if self.global_step % self.cfg_trainer.sh_degree_step_interval == 0:
            self.gaussians.step_sh_degree()

        return super().on_train_batch_start(batch, batch_idx)

    def training_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        data_dict = self.process_data(batch)
        loss, loss_log, results = self.calculate_loss(data_dict)

        self.manual_backward(loss)
        optimizer: torch.optim.Optimizer = self.optimizers()
        optimizer.step()
        optimizer.zero_grad()

        scheduler: torch.optim.lr_scheduler.LRScheduler = self.lr_schedulers()
        scheduler.step()

        self.cached_results = {
            "radii2D": results["radii2D"].detach(),
            "screenspace_gradient": results["screenspace_points"].grad.detach(),
        }

        self.log_dict(loss_log, prog_bar=True, logger=False, on_step=True)
        self.log_dict({f"train_{k}": v for k, v in loss_log.items()}, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("n_gaussians", len(self.gaussians.get_xyz()), prog_bar=False, logger=True, on_step=True, on_epoch=False)

        if self.global_step % self.cfg_trainer.display_interval == 0:
            mlflow_logger: MLFlowLogger = self.logger
            image = self.get_visuals(data_dict)
            mlflow_logger.log_image(image, f"train_image-{self.global_step:0>8d}.jpg", artifact_path="train_image")
            mlflow_logger.log_image(image, "train_image-latest.jpg")

        return loss

    def densify_gaussian(self, radii2D: torch.Tensor, screenspace_gradient: torch.Tensor) -> None:
        visible_mask = (radii2D > 0)
        self.gaussians.update_max_radii2D(radii2D, visible_mask)
        self.gaussians.update_xyz_gradient(screenspace_gradient, visible_mask)

        cfg = self.cfg_trainer
        if self.global_step > cfg.densify_since and self.global_step % cfg.densify_interval == 0:
            # add or remove gaussian according to the threshold
            preserve_idx = self.gaussians.densify_and_prune(
                densify_grad_threshold=cfg.densify_grad_threshold,
                clone_size_threshold=cfg.clone_size_threshold,
                prune_opacity_threshold=cfg.prune_opacity_threshold,
                prune_size_threshold=cfg.prune_size_threshold,
                prune_screensize_threshold=cfg.prune_screensize_threshold if self.global_step > cfg.opacity_reset_interval else None,
            )
            self.gaussians.reset_max_radii2D()
            self.gaussians.reset_xyz_gradient()

            # update the optimizer parameters
            self.update_optimizer_parameters(preserve_idx=preserve_idx)

    def update_optimizer_parameters(self, preserve_idx: List[int]):
        optimizer: torch.optim.Optimizer = self.optimizers()
        for group in optimizer.param_groups:
            name = group.get("name", None)
            if name is None or name not in self.gaussians.PARAMETER_NAMES:
                continue

            assert len(group["params"]) == 1

            new_params = getattr(self.gaussians, f"_{name}")
            stored_state = optimizer.state.get(group["params"][0], None)
            if stored_state is None:
                group["params"][0] = new_params
            else:
                diff_N = len(new_params) - len(preserve_idx)
                new_exp_avg = torch.zeros(diff_N, *stored_state["exp_avg"].shape[1:]).to(stored_state["exp_avg"])
                stored_state["exp_avg"] = torch.cat([stored_state["exp_avg"][preserve_idx], new_exp_avg], dim=0)
                new_exp_avg_sq = torch.zeros(diff_N, *stored_state["exp_avg_sq"].shape[1:]).to(stored_state["exp_avg_sq"])
                stored_state["exp_avg_sq"] = torch.cat([stored_state["exp_avg_sq"][preserve_idx], new_exp_avg_sq], dim=0)

                del optimizer.state[group["params"][0]]
                group["params"][0] = new_params
                optimizer.state[group["params"][0]] = stored_state

    def on_validation_epoch_start(self):
        if isinstance(self.trainer.val_dataloaders, list):
            min_n_batches = min(len(dataloader) for dataloader in self.trainer.val_dataloaders)
        else:
            min_n_batches = len(self.trainer.val_dataloaders)
        self.show_valid_idx = [0, np.random.randint(1, min_n_batches)]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        data_dict = self.process_data(batch)
        _, loss_log, results = self.calculate_loss(data_dict, calculate_screenspace_points=False)
        self.log_dict({f"valid_{k}": v for k, v in loss_log.items()}, prog_bar=False, logger=True)

        if batch_idx in self.show_valid_idx:
            mlflow_logger: MLFlowLogger = self.logger
            image = self.get_visuals(data_dict)
            prefix = "valid0_image" if batch_idx == 0 else "valid_image"
            mlflow_logger.log_image(image, f"{prefix}-{self.global_step:0>8d}.jpg", artifact_path=prefix)
            mlflow_logger.log_image(image, f"{prefix}-latest.jpg")

        return loss_log

    def on_validation_end(self):
        mlflow_logger: MLFlowLogger = self.logger
        mlflow_logger.log_object(
            obj=self.gaussians,
            obj_name=f"point_cloud-{self.global_step:0>8d}.ply",
            dump_func=lambda gs, p: gs.save_ply(p),
            artifact_path="point_cloud",
        )
        return super().on_validation_end()

    def calculate_loss(self, data_dict: dict, calculate_screenspace_points: bool = True) -> Tuple[torch.Tensor, dict, dict]:
        rendered_image, radii2D, depth_image, screenspace_points = self.render(
            data_dict,
            self.cfg_trainer.compute_cov3D_python,
            self.cfg_trainer.convert_SHs_python,
            self.cfg_trainer.seperate_SHs,
            calculate_screenspace_points=calculate_screenspace_points,
        )

        loss_recon = self.criterion_recon(rendered_image.unsqueeze(0), data_dict["image"].unsqueeze(0))
        loss_ssim = self.criterion_ssim(rendered_image.unsqueeze(0), data_dict["image"].unsqueeze(0))

        loss = (
            loss_recon * (1 - self.cfg_trainer.w_ssim) +
            loss_ssim * self.cfg_trainer.w_ssim
        )

        loss_log = dict(
            loss=loss.detach(),
            recon=loss_recon.detach(),
            ssim=loss_ssim.detach(),
        )

        results = dict(
            rendered=rendered_image,
            radii2D=radii2D,
            depth_image=depth_image,
            screenspace_points=screenspace_points,
        )

        return loss, loss_log, results

    def render(
        self,
        data_dict: dict,
        compute_cov3D_python: bool = False,
        convert_SHs_python: bool = False,
        seperate_SHs: bool = False,
        calculate_screenspace_points: bool = True,  # record gradients of the 2D(screen-space) means
    ):
        if compute_cov3D_python:
            raise NotImplementedError()
        if convert_SHs_python:
            raise NotImplementedError()
        if seperate_SHs:
            raise NotImplementedError()

        if calculate_screenspace_points:
            screenspace_points = torch.zeros_like(self.gaussians.get_xyz(), requires_grad=True)
        else:
            screenspace_points = None

        H, W = data_dict["image"].shape[-2:]
        raster_settings = GaussianRasterizationSettings(
            image_height=H,
            image_width=W,
            tanfovx=data_dict["tanfovx"],
            tanfovy=data_dict["tanfovy"],
            bg=data_dict["background"],
            scale_modifier=1.0,
            viewmatrix=data_dict["viewmatrix"],
            projmatrix=data_dict["projmatrix"],
            sh_degree=self.gaussians.active_sh_degree,
            campos=data_dict["campos"],
            prefiltered=False,
            debug=False,
            antialiasing=False
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        rendered_image, radii2D, depth_image = rasterizer(
            means3D=self.gaussians.get_xyz(),
            means2D=screenspace_points,
            shs=self.gaussians.get_features(),
            colors_precomp=None,
            opacities=self.gaussians.get_opacity(),
            scales=self.gaussians.get_scaling(),
            rotations=self.gaussians.get_rotation(),
            cov3D_precomp=None,
        )

        return rendered_image, radii2D, depth_image, screenspace_points

    @torch.no_grad()
    def get_visuals(self, data_dict) -> torch.Tensor:
        rendered_image, radii2D, depth_image, screenspace_points = self.render(
            data_dict,
            self.cfg_trainer.compute_cov3D_python,
            self.cfg_trainer.convert_SHs_python,
            self.cfg_trainer.seperate_SHs,
            calculate_screenspace_points=False,
        )
        image = data_dict["image"]
        vis = torch.cat([image, rendered_image, depth_image.expand_as(image)], -1)
        return vis
