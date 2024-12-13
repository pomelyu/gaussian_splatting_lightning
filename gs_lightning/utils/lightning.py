import tempfile
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Union

import numpy as np
import torchvision
from mlflow import MlflowClient
from pytorch_lightning.loggers import MLFlowLogger as _MLFlowLogger
from torch import Tensor

from .image import save_image


class MLFlowLogger(_MLFlowLogger):
    @property
    def has_tmp_dir(self) -> bool:
        return hasattr(self, "_tmp_dir")

    @property
    def tmp_dir(self) -> Path:
        if not self.has_tmp_dir:
            self._tmp_dir_obj = tempfile.TemporaryDirectory()
            self._tmp_dir = Path(self._tmp_dir_obj.name).resolve()
            assert self._tmp_dir.is_dir()
        return self._tmp_dir

    def log_artifact(self, local_path: str, artifact_path: str = None):
        mlflow_clent: MlflowClient = self.experiment
        mlflow_clent.log_artifact(self.run_id, local_path, artifact_path=artifact_path)

    def log_object(self, obj: Any, obj_name: str, dump_func: Callable, artifact_path: str = None) -> None:
        local_path = self.tmp_dir / obj_name
        dump_func(obj, local_path)
        self.log_artifact(local_path, artifact_path)

    def log_image(self, image: Union[Tensor, np.ndarray], image_name: str, artifact_path: str = None) -> None:
        if isinstance(image, Tensor):
            dump_func = torchvision.utils.save_image
        elif isinstance(image, np.ndarray):
            dump_func = save_image
        else:
            raise NotImplementedError(f"Unsupport format: {type(image)}")

        self.log_object(image, image_name, dump_func, artifact_path)

    def __del__(self):
        if self.has_tmp_dir:
            self._tmp_dir_obj.cleanup()
            print("cleanup logger temporary directory")
