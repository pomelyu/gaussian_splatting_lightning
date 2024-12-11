from typing import Union

import mlconfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler


class ConfigTrainDataloader(DataLoader):
    def __init__(self, dataset: Union[DictConfig, Dataset], num_iters: int, batch_size = 1, num_workers = 0, pin_memory = False):
        if isinstance(dataset, DictConfig):
            dataset = mlconfig.instantiate(dataset)
        elif isinstance(dataset, Dataset):
            pass
        else:
            raise TypeError(f"Unsupport dataset format: {type(dataset)}")

        sampler = RandomSampler(dataset, num_samples=num_iters * batch_size)
        super().__init__(dataset, batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)


class ConfigValidDataloader(DataLoader):
    def __init__(self, dataset: Union[DictConfig, Dataset], batch_size = 1, num_workers = 0, pin_memory = False):
        if isinstance(dataset, DictConfig):
            dataset = mlconfig.instantiate(dataset)
        elif isinstance(dataset, Dataset):
            pass
        else:
            raise TypeError(f"Unsupport dataset format: {type(dataset)}")

        sampler = SequentialSampler(dataset)
        super().__init__(dataset, batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
