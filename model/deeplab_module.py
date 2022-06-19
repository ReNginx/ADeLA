import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics import IoU
from torch import nn

from model.deeplab import Res_Deeplab
from model.loss import SoftCrossEntropy
from utils.image_utils import ImageUtils


class Deeplab(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.save_hyperparameters()
        self.model = Res_Deeplab()

        self.cross_entropy = SoftCrossEntropy()
        self.hard_cross_entropy = nn.CrossEntropyLoss()
        self.iou = IoU(num_classes=40, dist_sync_on_step=False, compute_on_step=False, reduction='none')

        if self.hparams.pretrain is not None:
            pretrain_ckpt = torch.load(self.hparams.pretrain)
            self.model.load_state_dict(pretrain_ckpt, strict=False)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--init_lr', type=float, default=1e-2)
        parser.add_argument('--lr_reduce_factor', type=float, default=0.5)
        parser.add_argument('--log_image_every_n_steps', type=int, default=100)
        parser.add_argument('--log_files', type=str, default='log.txt')
        parser.add_argument('--save_samples', action='store_true')
        parser.add_argument('--test_period', type=int, default=10)
        parser.add_argument('--n_channel', type=int, default=128)
        parser.add_argument('--target_angles', nargs='+', type=str, default=['30'])
        parser.add_argument('--pretrain', type=str, default=None)
        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.optim_parameters(self.hparams.init_lr), lr=self.hparams.init_lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.5)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": 'val_loss'}

    @property
    def need_test_current_epoch(self):
        return self.current_epoch > 0 and (self.current_epoch + 1) % self.hparams.test_period == 0

    def step(self, batch, idx, mode='train'):
        rgb, hard_label, soft_label, pseudo = batch

        pred = self.model(rgb)

        pred_label = pred.argmax(dim=1)
        hard_label = hard_label.squeeze(dim=1)

        log_step = self.global_step + (0 if mode == 'train' else idx)

        if log_step % self.hparams.log_image_every_n_steps == 0:
            if mode == 'train' and soft_label.max() > 0:
                pseudo = pseudo.squeeze(dim=1)
                grid = ImageUtils.make_grid_with_first_image([
                    rgb,
                    ImageUtils.labels_to_tensor(pred_label),
                    ImageUtils.labels_to_tensor(hard_label),
                    ImageUtils.labels_to_tensor(pseudo),
                ], [
                    'rgb', 'pred', 'semantics', 'pseudo'
                ])
                self.logger.experiment.add_image(f'{mode}_vis', grid, log_step)
            else:
                grid = ImageUtils.make_grid_with_first_image([
                    rgb,
                    ImageUtils.labels_to_tensor(pred_label),
                    ImageUtils.labels_to_tensor(hard_label),
                ], [
                    'rgb', 'pred', 'semantics'
                ])
                self.logger.experiment.add_image(f'{mode}_vis', grid, log_step)

        if mode != 'train':
            dummy_loss = torch.Tensor(0).to(pred_label.device)
            self.log(f'{mode}_loss', dummy_loss)

            if mode == 'test' or self.need_test_current_epoch:
                self.iou = self.iou.cpu()
                pred_label_cpu = pred_label.cpu()
                hard_label_cpu = hard_label.cpu()
                self.iou(pred_label_cpu, hard_label_cpu)
            return dummy_loss

        if mode == 'train':
            loss = 0
            for (p, h, s) in zip(pred, hard_label, soft_label):
                p = p.unsqueeze(0)
                h = h.unsqueeze(0)
                s = s.unsqueeze(0)
                if s.max() > 0:
                    loss += self.cross_entropy(p, s)  # n, c, h, w
                else:
                    loss += self.hard_cross_entropy(p, h)
            loss = loss / pred.shape[0]
            self.log('loss', loss)
            return loss

    def training_step(self, train_batch, batch_idx):
        return self.step(train_batch, batch_idx, mode='train')

    def validation_step(self, val_batch, batch_idx):
        return self.step(val_batch, batch_idx, mode='val')

    def test_step(self, test_batch, batch_idx):
        return self.step(test_batch, batch_idx, mode='test')

    def get_info(self, iou, extra_info=''):
        mIoU = iou.mean().detach().cpu().float() * 100
        iou = [round(scr * 100, 2) for scr in iou.detach().cpu().tolist()]
        info = f'''
============================     begin  {extra_info}@epoch:{self.current_epoch:02d}   ===========================
Mean IoU {mIoU:.2f}
per class IoU {iou}
============================     end  {extra_info}    ===========================
                    '''
        return info

    def calc_and_print_result(self, file_name):
        iou_res = self.iou.compute()

        if self.global_rank == 0:
            with open(os.path.join(self.logger.log_dir, file_name), 'w') as f:
                info = self.get_info(iou_res)
                print(info)
                f.write(info)

        self.iou.reset()

    def validation_epoch_end(self, outputs):
        if not self.need_test_current_epoch:
            return

        file_name = f'test@{self.current_epoch:02d}.txt'
        self.calc_and_print_result(file_name)

    def test_epoch_end(self, test_step_output):
        self.calc_and_print_result(f'test_res.txt')
