import argparse

import mlconfig
from mlconfig.torch import register_torch_optimizers
from mlconfig.torch import register_torch_schedulers
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything

import gs_lightning
from gs_lightning.utils.lightning import MLFlowLogger
from gs_lightning.utils.sys import get_current_time
from gs_lightning.utils.sys import get_current_utc_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    register_torch_optimizers()
    register_torch_schedulers()

    config = mlconfig.load(args.config)

    # Setup logger
    mlflow_logger = MLFlowLogger(
        experiment_name=config.exp_name,
        run_name=config.run_name,
        save_dir=config.mlflow_folder
    )
    mlflow_logger.log_artifact(args.config)

    # Setup trainer
    trainer = Trainer(
        logger=mlflow_logger,
        max_steps=config.cfg_trainer.num_iters,
        val_check_interval=config.cfg_trainer.valid_interval,
        log_every_n_steps=config.cfg_trainer.print_interval,
        enable_checkpointing=False,
        use_distributed_sampler=False,
    )

    # Setup meta
    meta = {
        "exp_id": mlflow_logger.experiment_id,
        "run_id": mlflow_logger.run_id,
        "timestamp": get_current_time(),
        "utc_time": get_current_utc_time(),
        "random_seed": args.seed,
    }

    for key, value in meta.items():
        print(f"{key}:", value)

    # Setup data
    if hasattr(config, "data"):
        datamodule = mlconfig.instantiate(config.data)
        cfg_train_dataloader = None
        cfg_valid_dataloader = None
    else:
        datamodule = None
        cfg_train_dataloader = config.cfg_train_dataloader
        cfg_valid_dataloader = config.cfg_valid_dataloader

    # Setup module
    model_class = mlconfig.getcls({"name": config.trainer})
    model = model_class(
        meta=meta,
        cfg_trainer=config.cfg_trainer,
        cfg_model=config.cfg_model,
        cfg_optimizer=config.cfg_optimizer,
        cfg_scheduler=config.cfg_scheduler,
        cfg_train_dataloader=cfg_train_dataloader,
        cfg_valid_dataloader=cfg_valid_dataloader,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
