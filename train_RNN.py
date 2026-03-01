import argparse
from typing import Optional

import torch
import torchvision.transforms as tvt
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from torch.utils.data import DataLoader
from model.dete_model.SPTransformer import Model
from dataload.dataset import SolarDataModule
torch.set_float32_matmul_precision('high')


def main(args):

    # W&B logger
    logger = WandbLogger(project=args.wandb_project_name,
                         group=args.wandb_group,
                         id=args.wandb_id)
    logger.log_hyperparams(vars(args))


    config_path = "/home/joty/code/FM_interpolation/config/config.yaml"
    dm = SolarDataModule(config_path)
    dm.prepare_data() 
    dm.setup(stage='fit') 

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=50,
        min_delta=1e-6,
        verbose=True
    )
    ckpt_callback = ModelCheckpoint(
        dirpath=args.checkpoint_path,
        filename="{epoch}-{val_loss:.6f}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        save_last=True,
        every_n_epochs=1
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')

    # Trainer
    trainer = Trainer(
        logger=logger,
        max_epochs=args.max_epochs,
        accelerator='gpu',
        strategy='auto',
        devices=args.num_gpus,
        callbacks=[ckpt_callback, lr_monitor_callback, early_stop_callback],
        precision=args.precision,
        check_val_every_n_epoch=args.check_val_every_n_epoch
    )

    # Model
    model = Model(
    scaler=dm.scaler,
    test_results_path=args.output_dir     
    )
    train = False
    if train:
        trainer.fit(model, dm)
    else: 
        test_model = Model.load_from_checkpoint(
         checkpoint_path=args.ckpt_path,
         scaler=dm.scaler,
         test_results_path=args.output_dir
        )
        dm.setup(stage='test')
        trainer.test(test_model, dm.test_dataloader())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Precision
    parser.add_argument('--precision', type=str, required=False,
                        choices=['bf16', '32', 'bf16-mixed'],
                        default='bf16',
                        help='The precision used for training.')
    parser.add_argument('--output_dir', type=str, default='/home/joty/code/FM_interpolation/outputs/test_infer',
                        help='Directory to save inference results (used by model.test_step).')
    # Optimizer
    parser.add_argument('--lr', type=float, required=False, default=5e-4,
                        help='Optimizer learning rate.')

    # Training schedule
    parser.add_argument('--max_epochs', type=int, required=False, default=500,
                        help='Number of training epochs.')
    parser.add_argument('--num_gpus', type=int, required=False, default=1,
                        help='Number of gpus to use.')
    parser.add_argument('--check_val_every_n_epoch', type=int, required=False, default=1,
                        help='Check validation every n epochs.')

    #  checkpoints
    parser.add_argument('--checkpoint_path', type=str, required=False, default='/home/joty/code/FM_interpolation/checkpoints/RNN',
                        help='Directory to save training checkpoints.')
    parser.add_argument('--ckpt_path', type=str, required=False, default='/home/joty/code/FM_interpolation/checkpoints/RNN',
                        help='Directory to save training checkpoints.')
    parser.add_argument('--resume_from_ckpt', type=str, required=False, default=None,
                        help='Resume lightning training from this checkpoint path.')

    # Logging
    parser.add_argument('--wandb_project_name', type=str, required=True, default='RNN',
                        help='Project name for Weights & Biases logger.')
    parser.add_argument('--wandb_group', type=str, required=False, default=None,
                        help='Group name for W&B runs.')
    parser.add_argument('--wandb_id', type=str, required=False, default=None,
                        help='Run id to resume W&B logging.')

    args = parser.parse_args()
    main(args)