# -*-coding:utf-8-*-
# -*-coding:utf-8-*-
import os

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
from monai.losses import DiceLoss, DiceFocalLoss, DiceCELoss, FocalLoss, GeneralizedDiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from monai.data import decollate_batch, list_data_collate
from monai.networks.nets import UNETR, UNet, VNet, DynUNet, AttentionUnet, SwinUNETR
from timm.models.layers import trunc_normal_
from monai.data import NiftiSaver, PNGSaver, ImageWriter
from monai.transforms import *
from monai.inferers.inferer import sliding_window_inference, SliceInferer

from TransUnet.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


class Config():
    '''
    {data_dir}
        Testing data
        Training data
    '''
    data_dir = r'D:\Caiyimin\Dataset\Small3D\KeDaXunFei2022BrainVessel'  # 指定数据集路径，路径格式如下：
    seed = 42

    Final_shape = [448, 448, 128]
    # Final_shape = [224, 224, 128]
    AxCode = 'RAS'
    Spacing = [1.0268, 1.0268, 0.8]

    train_ratio, val_ratio, test_ratio = [0.9, 0.1, 0.0]
    lowerPercent = 0.5
    upperPercent = 99.5

    # HuMin = 0.0
    # HuMax = 255.0

    in_channels = 1
    BatchSize = 1
    RealBatchSize = 4
    SW_batch_size = 32  # 滑动窗口推理中的batch_size
    lr = 3e-4  # 学习率
    NumWorkers = 4

    n_classes = 1  # 背景(0) + 血管
    UseSigmoid = False  # 针对二分类是使用sigmoid函数还是softmax函数
    if (UseSigmoid and n_classes >= 2) or (not UseSigmoid and n_classes == 1):
        raise ("Config Fatal")

    max_epoch = 1000
    min_epoch = 60

    ModelDict = {}
    ArgsDict = {}
    TypeDict = {}

    model_name = 'Unet3D'
    # model_name = 'VNet'p+0./    # model_name = "AttentionUnet3D"

    # model_name = 'Unet'

    # 爆显存的模型
    # model_name = 'UNetR'
    # model_name = "TransUnet"

    ModelDict['Unet3D'] = UNet
    ArgsDict['Unet3D'] = {'spatial_dims': 3, 'in_channels': in_channels, 'out_channels': n_classes,
                          'channels': (32, 64, 128, 256, 512), 'strides': (2, 2, 2, 2),
                          'num_res_units': 2, 'bias': True,  # 'dropout': 0.2,
                          }
    TypeDict['Unet3D'] = '3D'

    ModelDict['Unet'] = UNet
    ArgsDict['Unet'] = {'spatial_dims': 2, 'in_channels': in_channels, 'out_channels': n_classes,
                        'channels': (32, 64, 128, 256, 512), 'strides': (2, 2, 2, 2),
                        'num_res_units': 2, 'bias': True, 'dropout': 0.2,
                        }
    TypeDict['Unet'] = '2D'

    ModelDict['VNet'] = VNet
    ArgsDict['VNet'] = {'spatial_dims': 3, 'in_channels': in_channels, 'out_channels': n_classes, 'dropout_prob': 0.2, }
    TypeDict['VNet'] = '3D'

    ModelDict['AttentionUnet3D'] = AttentionUnet
    ArgsDict['AttentionUnet3D'] = {'spatial_dims': 3, 'in_channels': in_channels, 'out_channels': n_classes,
                                   'channels': (32, 64, 128, 256, 512), 'strides': (2, 2, 2, 2), }
    TypeDict['AttentionUnet3D'] = '3D'

    ModelDict['UNetR'] = UNETR
    ArgsDict['UNetR'] = {'in_channels': in_channels, 'out_channels': n_classes, 'img_size': Final_shape}
    TypeDict['UNetR'] = '3D'

    vit_name = 'R50-ViT-B_16'
    vit_patches_size = 16
    img_size = Final_shape[0]
    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = n_classes
    config_vit.n_skip = 3
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    ModelDict['TransUnet'] = ViT_seg
    ArgsDict['TransUnet'] = {'config': config_vit, 'img_size': Final_shape[0], 'num_classes': n_classes}
    TypeDict['TransUnet'] = '2D'

    NeedTrain = True
    # NeedTrain = False  # 控制直接预测 还是 先训练再预测
    ValidSegDir = os.path.join(data_dir, 'ValidSeg', model_name)
    PredSegDir = os.path.join(data_dir, 'PredSeg', model_name, "labels")


class BrainVesselDataset(pl.LightningDataModule):
    def __init__(self, cfg=Config()):
        super(BrainVesselDataset, self).__init__()

        self.cfg = cfg
        self.img_dir = os.path.join(cfg.data_dir, r'Training data/Tran-MRA')
        self.label_dir = os.path.join(cfg.data_dir, r'Training data/Tran-MRA-labels')
        self.pred_dir = os.path.join(cfg.data_dir, r'Testing data/Test-MRA')
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
                                                                      generator=torch.Generator().manual_seed(cfg.seed)
                                                                      )

    def get_init(self):
        train_labels = sorted(glob(os.path.join(self.label_dir, "*.mhd")))
        train_imgs = sorted(glob(os.path.join(self.img_dir, "*.mhd")))
        self.pred_dict = sorted(glob(os.path.join(self.pred_dir, "*.mhd")))
        assert len(train_imgs) == len(train_labels)
        return train_imgs, train_labels

    def get_preprocess(self):  # 控制预处理相关的transform
        cfg = self.cfg
        self.train_process = Compose([
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(keys=['image', 'label']),

            Orientationd(keys=['image', 'label'], axcodes=cfg.AxCode),
            # Spacingd(keys=['image', 'label'], pixdim=cfg.Spacing),

            # RandFlipd(keys=['image', 'label'], prob=0.25, spatial_axis=[0]),
            # RandFlipd(keys=['image', 'label'], prob=0.25, spatial_axis=[1]),
            # RandFlipd(keys=['image', 'label'], prob=0.25, spatial_axis=[2]),
            # RandRotate90d(keys=['image', 'label'], prob=0.25),
            # RandAffined(keys=['image', 'label'], spatial_size=cfg.Final_shape, prob=0.1, mode=['bilinear', 'nearest']),

            # RandAdjustContrastd(keys=['image'], prob=0.25),

            NormalizeIntensityd(keys=['image'], ),

            EnsureTyped(keys=['image', 'label']),
        ])

        self.val_process = Compose([
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(keys=['image', 'label']),

            Orientationd(keys=['image', 'label'], axcodes=cfg.AxCode),
            # Spacingd(keys=['image', 'label'], pixdim=cfg.Spacing),

            NormalizeIntensityd(keys=['image'], ),
            EnsureTyped(keys=['image', 'label']),
        ])

        self.pred_process = Compose([
            LoadImaged(keys=['image']),
            EnsureChannelFirstd(keys=['image']),

            Orientationd(keys=['image', 'label'], axcodes=cfg.AxCode),
            # Spacingd(keys=['image', 'label'], pixdim=cfg.Spacing),

            NormalizeIntensityd(keys=['image'], ),
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
            self.loss_func = DiceLoss(include_background=True, squared_pred=False, sigmoid=True)
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
            self.loss_func = DiceLoss(include_background=False, softmax=True,
                                      to_onehot_y=True, squared_pred=False)
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

        self.best_dice = 0.0

    def configure_optimizers(self):
        cfg = self.cfg
        opt = optim.AdamW(params=self.net.parameters(),
                          lr=cfg.lr, eps=1e-7,
                          weight_decay=1e-5)

        # opt = optim.SGD(params=self.net.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=1e-5)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, threshold=1e-5, patience=5)

        if cfg.val_ratio > 0.0:
            return {'optimizer': opt, 'lr_scheduler': lr_scheduler, 'monitor': 'valid_mean_loss'}
        else:
            return {'optimizer': opt, 'lr_scheduler': lr_scheduler, 'monitor': 'train_mean_loss'}

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
        meta_dict = batch['image_meta_dict']  # 将meta_dict中的值转成cpu()向量，原来位于GPU上
        self.net.eval()

        if cfg.TypeDict[cfg.model_name] == '2D':
            y_pred = self.sliceInfer(x, self.net)
        else:
            y_pred = self.net(x)

        if not cfg.UseSigmoid:
            y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
        else:
            y_pred = self.post_pred(y_pred)
        self.save_pred(meta_dict, y_pred)
        return y_pred, meta_dict

    def save_pred(self, meta_dict, y_pred):
        for k, v in meta_dict.items():
            if isinstance(v, torch.Tensor):
                meta_dict[k] = v.detach().cpu()
        saver = NiftiSaver(output_dir=self.cfg.PredSegDir, mode="nearest", print_log=True,
                           separate_folder=False, output_postfix="")  # 禁止为单个文件新建目录
        saver.save_batch(y_pred, meta_dict)  # fixme 检查此处用法是否正确

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
            # trunc_normal_(m.weight, std=0.02)
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
        # accumulate_grad_batches=4, # 慎用，否则log里面显示出来的train_mean_loss会有很大的问题
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
    for file in ckptFiles:
        score = file.split("=")[-1][:-5]
        score = float(score)
        if bestScore < score:
            bestFile, bestScore = file, score

    model = BrainVessel.load_from_checkpoint(bestFile)  # 这是个类方法，不是对象方法
    model.eval()
    model.freeze()

    trainer.predict(model, datamodule=data)


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


if __name__ == '__main__':
    setseed()
    main()
    nii_gz2mhd()
