# -*-coding:utf-8-*-
# -*-coding:utf-8-*-
import os
import time

import torch
from torch import nn, functional as F, optim
import monai
import pytorch_lightning as pl
from monai.data import Dataset, SmartCacheDataset
from torch.utils.data import DataLoader, random_split
from glob import glob
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from monai.config import KeysCollection
from torch.utils.data import random_split
from monai.losses import DiceLoss, DiceFocalLoss, DiceCELoss, GlobalMutualInformationLoss, MaskedDiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from monai.data import decollate_batch, list_data_collate
from monai.networks.nets import UNETR, UNet, VNet, DynUNet, AttentionUnet, SwinUNETR
from timm.models.layers import trunc_normal_
from monai.data import NiftiSaver, PNGSaver, ImageWriter
from monai.transforms import *
from monai.inferers.inferer import sliding_window_inference, SliceInferer
import SimpleITK as stk

from TransUnet.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

from Config import Config


class BrainVesselDataset(pl.LightningDataModule):
    def __init__(self, cfg=Config()):
        super(BrainVesselDataset, self).__init__()

        self.cfg = cfg
        self.img_dir = os.path.join(cfg.data_dir, r'TrainingDataRemoveBone/Tran-MRA')
        self.label_dir = os.path.join(cfg.data_dir, r'TrainingDataRemoveBone/Tran-MRA-labels')
        self.pred_dir = os.path.join(cfg.data_dir, r'TestingDataRemoveBone/Test-MRA')
        self.img_label = []

        self.train_dict, self.val_dict, self.test_dict = None, None, None
        self.pred_dict = None

        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.pred_set = None

        self.train_process = None
        self.val_process = None
        self.pred_process = None

    def prepare_data(self):
        imgs, labels = self.get_init()
        for x, y in zip(imgs, labels):
            info = {'image': x, 'label': y}
            self.img_label.append(info)

        pred_dict = []
        for x in self.pred_dict:
            pred_dict.append({'image': x})
        self.pred_dict = pred_dict

        self.split_dataset(self.img_label)
        self.get_preprocess()

    def setup(self, stage=None):
        self.train_set = Dataset(self.train_dict, transform=self.train_process)
        self.val_set = Dataset(self.val_dict, transform=self.val_process)
        self.test_set = Dataset(self.test_dict, transform=self.val_process)
        self.pred_set = Dataset(self.pred_dict, transform=self.pred_process)

    def train_dataloader(self):
        cfg = self.cfg
        return DataLoader(self.train_set, batch_size=cfg.BatchSize,
                          num_workers=cfg.NumWorkers, collate_fn=list_data_collate,
                          shuffle=True)

    def val_dataloader(self):
        cfg = self.cfg
        return DataLoader(self.val_set, batch_size=cfg.BatchSize,
                          num_workers=cfg.NumWorkers)

    def test_dataloader(self):
        cfg = self.cfg
        return DataLoader(self.test_set, batch_size=cfg.BatchSize,
                          num_workers=cfg.NumWorkers)

    def predict_dataloader(self):
        cfg = self.cfg
        return DataLoader(self.pred_set, batch_size=cfg.BatchSize,
                          num_workers=cfg.NumWorkers)

    def split_dataset(self, img_label):
        cfg = self.cfg
        num = len(img_label)
        train_num = int(num * cfg.train_ratio)
        val_num = int(num * cfg.val_ratio)
        test_num = int(num * cfg.test_ratio)
        if train_num + val_num + test_num != num:
            remain = num - train_num - test_num - val_num
            val_num += remain

        self.train_dict, self.val_dict, self.test_dict = random_split(img_label, [train_num, val_num, test_num],
                                                                      # generator=torch.Generator().manual_seed(cfg.seed),
                                                                      generator=torch.Generator().manual_seed(
                                                                          int(time.time()))
                                                                      )

    def get_init(self):
        train_labels = sorted(glob(os.path.join(self.label_dir, "*.mhd")))
        train_imgs = sorted(glob(os.path.join(self.img_dir, "*.nii.gz")))
        self.pred_dict = sorted(glob(os.path.join(self.pred_dir, "*.nii.gz")))
        assert len(train_imgs) == len(train_labels)
        return train_imgs, train_labels

    def get_preprocess(self):  # 控制预处理相关的transform
        cfg = self.cfg
        self.train_process = Compose([
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(keys=['image', 'label']),

            Orientationd(keys=['image', 'label'], axcodes=cfg.AxCode),
            # Spacingd(keys=['image', 'label'], pixdim=cfg.Spacing),

            NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),

            # 这些变换参考了nnunet，参见monai tutorials里面的DynUnet预处理
            # 随机缩放
            RandZoomd(keys=['image', 'label'], min_zoom=0.85, max_zoom=1.25, mode=('trilinear', 'nearest'),
                      prob=0.15),
            # RandGaussianNoised(keys='image', std=0.01, prob=0.15),
            # RandGaussianSmoothd(keys=['image'], sigma_x=[0.5, 1.15], sigma_y=[0.5, 1.15], sigma_z=[0.5, 1.15],
            #                     prob=0.15),
            # RandScaleIntensityd(keys='image', factors=0.3, prob=0.15),

            RandRotated(keys=['image', 'label'], range_x=cfg.RotateRange, range_y=cfg.RotateAngle,
                        range_z=cfg.RotateRange, prob=0.5, mode=('bilinear', 'nearest')),

            EnsureTyped(keys=['image', 'label']),
        ])

        self.val_process = Compose([
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(keys=['image', 'label']),

            Orientationd(keys=['image', 'label'], axcodes=cfg.AxCode),
            # Spacingd(keys=['image', 'label'], pixdim=cfg.Spacing),

            NormalizeIntensityd(keys=['image'], nonzero=True),

            EnsureTyped(keys=['image', 'label']),
        ])

        self.pred_process = Compose([
            LoadImaged(keys=['image']),
            EnsureChannelFirstd(keys=['image']),

            Orientationd(keys=['image'], axcodes=cfg.AxCode),
            # Spacingd(keys=['image', ], pixdim=cfg.Spacing),

            NormalizeIntensityd(keys=['image'], nonzero=True),

            EnsureTyped(keys=['image']),
        ])


class BrainVessel(pl.LightningModule):
    def __init__(self, cfg=Config()):
        super(BrainVessel, self).__init__()
        self.cfg = cfg
        model = cfg.ModelDict[cfg.model_name]
        kwargs = cfg.ArgsDict[cfg.model_name]
        self.net = model(**kwargs)
        ModelParamInit(self.net)

        self.sliceInfer = SliceInferer(
            roi_size=cfg.Final_shape[:-1],
            sw_batch_size=cfg.SW_batch_size,
            spatial_dim=2,
            cval=-1,
            # progress=True,
        )

        if cfg.UseSigmoid:  # 针对二分类是使用sigmoid函数还是softmax函数
            self.loss_func = DiceCELoss(include_background=True, squared_pred=False, sigmoid=True)
            # self.loss_func = GlobalMutualInformationLoss()
            # self.loss_func = MaskedDiceLoss(include_background=True, squared_pred=False, sigmoid=True)
            # self.loss_func = DiceFocalLoss(include_background=True, squared_pred=False, sigmoid=True)

            self.train_metric = DiceMetric(include_background=True, reduction='mean_batch', get_not_nans=True)
            self.valid_metric = DiceMetric(include_background=True, reduction='mean_batch', get_not_nans=True)
            self.post_pred = Compose([
                EnsureType(),
                Activations(sigmoid=True),
                AsDiscrete(threshold=0.5)
            ])
            self.post_label = Compose([
                EnsureType(),
            ])
        else:
            # self.loss_func = GlobalMutualInformationLoss()
            # self.loss_func = DiceFocalLoss(include_background=True, to_onehot_y=True,
            #                                squared_pred=False, softmax=True)
            self.loss_func = DiceCELoss(include_background=True, to_onehot_y=True,
                                        squared_pred=False, softmax=True)

            self.train_metric = DiceMetric(include_background=False, reduction='mean_batch', get_not_nans=True)
            self.valid_metric = DiceMetric(include_background=False, reduction='mean_batch', get_not_nans=True)
            self.post_pred = Compose([
                EnsureType(),
                AsDiscrete(argmax=True, to_onehot=cfg.n_classes)
            ])
            self.post_label = Compose([
                EnsureType(),
                AsDiscrete(to_onehot=cfg.n_classes)
            ])

    def configure_optimizers(self):
        cfg = self.cfg

        opt = optim.SGD(params=self.net.parameters(), lr=cfg.SGDLr, momentum=cfg.Momentum)
        # opt = optim.AdamW(params=self.net.parameters(), lr=cfg.AdamLr, eps=1e-7)
        # opt = optim.Adam(params=self.net.parameters(), lr=cfg.AdamLr, eps=1e-7)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, threshold=1e-5, patience=5)

        if cfg.val_ratio > 0.0:
            return {'optimizer': opt, 'lr_scheduler': lr_scheduler, 'monitor': 'valid_mean_loss'}
        else:
            return {'optimizer': opt, 'lr_scheduler': lr_scheduler, 'monitor': 'train_mean_loss'}

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        cfg = self.cfg
        x = batch['image']
        y = batch['label']
        if cfg.TypeDict[cfg.model_name] == '2D':
            y_hat = self.sliceInfer(x, self.net)
        else:
            y_hat = self.net(x)
        loss, dice = self.shared_step(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, batch_size=cfg.BatchSize)
        self.log('train_dice', dice, prog_bar=True, batch_size=cfg.BatchSize)
        return {'loss': loss, 'train_dice': dice}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.net.eval()
        cfg = self.cfg
        x = batch['image']
        y = batch['label'].float()

        if cfg.TypeDict[cfg.model_name] == '2D':
            y_hat = self.sliceInfer(x, self.net)
        else:
            y_hat = self.net(x)
        # 使用滑动窗口进行推理
        # y_hat = sliding_window_inference(x, roi_size=cfg.Final_shape, sw_batch_size=1,
        #                                  predictor=self.net)

        loss, dice = self.shared_step(y_hat, y, is_train=False)
        self.log('valid_loss', loss, prog_bar=True, batch_size=cfg.BatchSize)
        self.log('valid_dice', dice, prog_bar=True, batch_size=cfg.BatchSize)
        return {'valid_loss': loss, 'valid_dice': dice}

    def predict_step(self, batch, batch_idx):
        cfg = self.cfg
        x = batch['image']
        meta_dict = batch['image_meta_dict']
        self.net.eval()

        if cfg.TypeDict[cfg.model_name] == '2D':
            y_pred = self.sliceInfer(x, self.net)
        else:
            y_pred = self.net(x)

        return y_pred, meta_dict

    def training_epoch_end(self, outputs):
        losses, mean_dice = self.shared_epoch_end(outputs, 'loss', 'train_dice')
        if len(losses) > 0:
            mean_loss = torch.mean(losses)
            mean_loss = mean_loss.item()
            self.log('train_mean_loss', mean_loss, prog_bar=True)
            self.log('train_mean_dice', mean_dice[0], prog_bar=True)

    def validation_epoch_end(self, outputs):
        losses, mean_dice = self.shared_epoch_end(outputs, 'valid_loss', is_train=False)
        if len(losses) > 0:
            mean_loss = torch.mean(losses)
            mean_loss = mean_loss.item()
            self.log('valid_mean_loss', mean_loss, prog_bar=True)
            self.log('valid_mean_dice', mean_dice[0], prog_bar=True)

    def shared_epoch_end(self, outputs, loss_key, is_train: bool = True):
        losses = []
        for output in outputs:
            # loss = output['loss'].detach().cpu().numpy()
            loss = output[loss_key]
            loss = loss.detach()
            losses.append(loss)

        losses = torch.stack(losses)

        if is_train:
            mean_dice = self.train_metric.aggregate()
            self.train_metric.reset()
        else:
            mean_dice = self.valid_metric.aggregate()
            self.valid_metric.reset()
        return losses, mean_dice

    def shared_step(self, y_hat, y, is_train: bool = True):
        loss = self.loss_func(y_hat, y)
        y_hat = [self.post_pred(it) for it in decollate_batch(y_hat)]
        y = [self.post_label(it) for it in decollate_batch(y)]

        if is_train:
            dice = self.train_metric(y_pred=y_hat, y=y)
            dice = torch.mean(dice, dim=0)
        else:
            dice = self.valid_metric(y_pred=y_hat, y=y)
            dice = torch.mean(dice, dim=0)
        return loss, dice


def ModelParamInit(model):
    assert isinstance(model, nn.Module)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Conv3d, nn.ConvTranspose3d)):
            trunc_normal_(m.weight, std=0.02)
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm3d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def main():
    data = BrainVesselDataset()
    model = BrainVessel()

    cfg = Config()

    if cfg.val_ratio > 0.0:
        monitor = "valid_mean_loss"
        filename = '{epoch}-{valid_mean_loss:.4f}-{valid_mean_dice:.4f}'
    else:
        monitor = "train_mean_loss"
        filename = '{epoch}-{train_mean_loss:.4f}-{train_mean_dice:.4f}'

    early_stop = EarlyStopping(
        monitor=monitor,
        patience=20,
    )
    check_point = ModelCheckpoint(dirpath=f'./logs/{cfg.model_name}',
                                  save_last=False,
                                  save_top_k=3, monitor=monitor, verbose=True,
                                  filename=filename)
    trainer = pl.Trainer(
        progress_bar_refresh_rate=10,
        max_epochs=cfg.max_epoch,
        min_epochs=cfg.min_epoch,
        gpus=1,
        auto_scale_batch_size=True,  # 这个参数对模型训练速度很关键
        # accumulate_grad_batches=2,  # 慎用，否则log里面显示出来的train_mean_loss会有很大的问题
        logger=TensorBoardLogger(save_dir=f'./logs', name=f'{cfg.model_name}'),
        callbacks=[early_stop, check_point],
        precision=16,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        gradient_clip_val=1e3,
        gradient_clip_algorithm='norm',
    )

    if Config.NeedTrain:
        trainer.fit(model, data)
        trainer.save_checkpoint(f'./trained_models/{cfg.model_name}/TrainedModel.ckpt')

    # 选择dice系数最好的ckpt
    ckptFiles = glob(os.path.join(f'./logs/{cfg.model_name}', "*.ckpt"))
    bestFile = ""
    bestScore = 0.0
    for info in ckptFiles:
        score = info.split("=")[-1][:-5]
        score = float(score)
        if bestScore < score:
            bestFile, bestScore = info, score

    model = BrainVessel.load_from_checkpoint(bestFile)  # 这是个类方法，不是对象方法
    model.eval()
    model.freeze()

    if not cfg.NeedTrain:
        data.prepare_data()
        data.setup()

    pred_dict = data.pred_dict
    pred_trans = data.pred_process
    for file in pred_dict:
        single_set = Dataset([file], transform=pred_trans)
        pred_dataloader = DataLoader(single_set, batch_size=1)
        info = trainer.predict(model, dataloaders=pred_dataloader, return_predictions=True)
        for (y_pred, meta_dict) in info:
            if not cfg.UseSigmoid:
                y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
            else:
                post_pred = Compose([
                    EnsureType(),
                    Activations(sigmoid=True),
                    AsDiscrete(threshold=0.5)
                ])
                y_pred = post_pred(y_pred)
            save_pred(meta_dict, y_pred)


def save_pred(meta_dict, y_pred):
    for k, v in meta_dict.items():
        if isinstance(v, torch.Tensor):
            meta_dict[k] = v.detach().cpu()
    saver = NiftiSaver(output_dir=Config.PredSegDir, mode="nearest", print_log=True,
                       separate_folder=False, output_postfix="")  # 禁止为单个文件新建目录
    saver.save_batch(y_pred, meta_dict)  # fixme 检查此处用法是否正确


def setseed(seed: int = 42):
    pl.seed_everything(seed)
    set_determinism(seed)


def nii_gz2mhd(path: str = os.path.join(Config.PredSegDir, )):
    import SimpleITK as stk
    files = glob(os.path.join(path, "*.nii.gz"))
    for file in files:
        img = stk.ReadImage(file)
        # .nii.gz
        save_path = file.split(".")[0]  # file[:-7]
        save_path = save_path + ".mhd"
        stk.WriteImage(img, save_path)

    for file in files:  # 删除nii.gz文件
        os.remove(file)


def mhd2nii(path: str = os.path.join(Config.PredSegDir, )):
    import SimpleITK as stk
    files = glob(os.path.join(path, "*.mhd"))
    for file in files:
        img = stk.ReadImage(file)
        # .nii.gz
        save_path = file.split(".")[0]
        save_path = save_path + ".nii.gz"
        stk.WriteImage(img, save_path)


def removePredSuffix(path: str = Config.PredSegDir):  # 去除_brain后缀
    files = glob(os.path.join(path, "*.nii.gz"))
    for file in files:
        new_name = file.split("_")[0] + ".nii.gz"

        # order = f"mv {file} {new_name}"
        # # print(order)
        # os.system(order)

        os.rename(file, new_name)


if __name__ == '__main__':
    setseed()
    main()
    removePredSuffix()
    nii_gz2mhd()
