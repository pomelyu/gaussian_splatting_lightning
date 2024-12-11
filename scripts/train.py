import argparse

import mlconfig
from mlconfig.torch import register_torch_optimizers
from mlconfig.torch import register_torch_schedulers
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import MLFlowLogger

import gs_lightning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    register_torch_optimizers()
    register_torch_schedulers()

    config = mlconfig.load(args.config)

    exp_logger = MLFlowLogger(
        experiment_name=config.exp_name,
        run_name=config.run_name,
        save_dir=config.mlflow_folder
    )

    trainer = Trainer(
        logger=exp_logger,
        max_steps=config.cfg_trainer.num_iters,
        val_check_interval=config.cfg_trainer.valid_interval,
        log_every_n_steps=config.cfg_trainer.print_interval,
        enable_checkpointing=False,
    )

    if hasattr(config, "data"):
        datamodule = mlconfig.instantiate(config.data)
        cfg_train_dataloader = None
        cfg_valid_dataloader = None
    else:
        datamodule = None
        cfg_train_dataloader = config.cfg_train_dataloader
        cfg_valid_dataloader = config.cfg_valid_dataloader

    model_class = mlconfig.getcls({"name": config.trainer})
    model = model_class(
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
