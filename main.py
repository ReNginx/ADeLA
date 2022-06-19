import os
import shutil
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

from dataset.data_module import DataModule
from model.adela_module import ADeLA
from model.deeplab_module import Deeplab

if __name__ == '__main__':
    parser = ArgumentParser(conflict_handler='resolve', allow_abbrev=False)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--dataset_type', type=str, default='')
    type_args, _ = parser.parse_known_args()

    parser.add_argument('--seed', type=int, default=233)
    parser.add_argument('--period', type=int, default=5)
    parser.add_argument('--version', type=str, default=None)
    parser.add_argument('--expr_name', type=str, default=None)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--ckpt_path', type=str, default=None)

    if type_args.model == 'adela':
        ModelType = ADeLA
    elif type_args.model == 'deeplab':
        ModelType = Deeplab
    else:
        raise NotImplementedError(f'Model {type_args.model_name} is not implemented.')

    parser = ModelType.add_model_specific_args(parser)
    parser = DataModule.add_argparse_args(parser, dataset_type=type_args.dataset_type)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='ckpt-{epoch:02d}-{val_loss:.2f}',
        save_top_k=-1,
        save_last=True,
        period=args.period
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_callback, lr_monitor]

    pl.seed_everything(args.seed)

    if args.version is not None:
        path = os.path.join('logs', args.expr_name, args.version)
        if os.path.isdir(path):
            shutil.rmtree(path)

    logger = TensorBoardLogger("logs", name=args.expr_name, version=args.version)
    modality_dm = DataModule(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, plugins=[DDPPlugin(find_unused_parameters=True)])
    if args.mode == 'train':
        model = ModelType(args)
        trainer.fit(model, datamodule=modality_dm)
    else:
        model = ModelType.load_from_checkpoint(args.ckpt_path, hparams=args)
        trainer.test(model, datamodule=modality_dm)
