import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


# https://github.com/sxyu/svox2/blob/ee80e2c4df8f29a407fda5729a494be94ccf9234/opt/util/util.py#L78
class GSWarmUpExponentialDecayScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        param: str,
        max_steps: int,
        lr_init: float,
        lr_final: float,
        lr_delay_multi: float = 1,
        lr_delay_step: int = 0,
    ):
        self.param = param
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.lr_delay_step = lr_delay_step
        self.lr_delay_multi = lr_delay_multi
        self.max_steps = max_steps

        super().__init__(optimizer, last_epoch=-1, verbose=False)

    def get_lr(self):
        current_step = self.last_epoch
        if self.lr_delay_step > 0:
            delay_rate = self.lr_delay_multi + (1 - self.lr_delay_multi) * np.sin(
                0.5 * np.pi * np.clip(current_step / self.lr_delay_step, 0, 1)
            )
        else:
            delay_rate = 1.0

        t = np.clip(current_step / self.max_steps, 0, 1)
        log_lerp = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        new_lr = delay_rate * log_lerp

        all_lrs = []
        for group in self.optimizer.param_groups:
            if group.get("name") == self.param:
                all_lrs.append(new_lr)
            else:
                all_lrs.append(group["lr"])
        return all_lrs
