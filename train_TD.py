# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import ArgumentParser
import os
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from datamodules import ArgoverseV2DataModule
from pipeline.hl_generation_pipeline import HLTrafficGen
from pipeline.infilling_pipeline import TrajInfill
from utils import parse_bool


os.environ['WANDB_IGNORE_GLOBS'] = '*.pt,*.pth,*.ckpt'

MODEL_CLASSES = {
    'hl': HLTrafficGen,
    'infill': TrajInfill,
}

CHECKPOINT_MONITORS = {
    'hl': 'val_offroad_rate',
    'infill': 'val_reg_loss_propose',
}

DATAMODULE_CLASSES = {
    'argoverse_v2': ArgoverseV2DataModule,
}


def next_version(folder):
    """Return the next numeric version for a log directory."""
    folder = Path(folder)
    versions = [
        int(path.name.removeprefix('version_'))
        for path in folder.iterdir()
        if path.is_dir()
        and path.name.startswith('version_')
        and path.name.removeprefix('version_').isdigit()
    ]
    return max(versions, default=0) + 1


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.set_defaults(dataset='argoverse_v2')
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--train_raw_dir', type=str, default=None)
    parser.add_argument('--val_raw_dir', type=str, default=None)
    parser.add_argument('--test_raw_dir', type=str, default=None)
    parser.add_argument('--train_processed_dir', type=str, default=None)
    parser.add_argument('--val_processed_dir', type=str, default=None)
    parser.add_argument('--test_processed_dir', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=str, default='1')
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--plot', type=parse_bool, default=False)
    parser.add_argument('--mode', choices=MODEL_CLASSES, default='infill')

    partial_args, _ = parser.parse_known_args()
    MODEL_CLASSES[partial_args.mode].add_model_specific_args(
        parser, training=True, prefix='')
    return parser


def main():
    pl.seed_everything(2024, workers=True)
    args = build_parser().parse_args()

    model = MODEL_CLASSES[args.mode](args)
    model.add_extra_param(args)
    datamodule = ArgoverseV2DataModule(**vars(args))

    checkpoint = ModelCheckpoint(
        monitor=CHECKPOINT_MONITORS[args.mode], save_top_k=5, mode='min')
    callbacks = [checkpoint, LearningRateMonitor(logging_interval='epoch')]

    experiment_dir = Path(f'logs_{args.mode}')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    version = next_version(experiment_dir)
    log_dir = experiment_dir / f'version_{version}'
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = WandbLogger(
        project='TD', log_model=True, name=f'version_{version}',
        save_dir=str(log_dir))

    trainer = pl.Trainer(
        logger=logger,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=DDPStrategy(
            find_unused_parameters=True, gradient_as_bucket_view=True),
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        num_sanity_val_steps=1,
        gradient_clip_val=1,
    )
    trainer.fit(model, datamodule)


if __name__ == '__main__':
    main()
