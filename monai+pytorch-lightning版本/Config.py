import os

import numpy as np
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
    OrgSpacing = [0.5134, 0.5134, 0.80001]

    train_ratio, val_ratio, test_ratio = [0.8, 0.2, 0.0]
    lowerPercent = 0.2
    upperPercent = 100

    HuMin = 200
    HuMax = 1000

    in_channels = 1
    BatchSize = 1
    RealBatchSize = 4
    SW_batch_size = 32  # 滑动窗口推理中的batch_size
    AdamLr = 3e-4  # 学习率
    SGDLr = 1e-2
    Momentum = 0.99
    NumWorkers = 4

    n_classes = 2  # 背景(0) + 血管
    UseSigmoid = False  # 针对二分类是使用sigmoid函数还是softmax函数
    if (UseSigmoid and n_classes >= 2) or (not UseSigmoid and n_classes == 1):
        raise ("Config Fatal")

    max_epoch = 500
    min_epoch = 100

    RotateAngle = 15.0 / 360.0 * 2 * np.pi
    RotateRange = [-RotateAngle, RotateAngle]

    ModelDict = {}
    ArgsDict = {}
    TypeDict = {}

    model_name = 'Unet3D'
    # model_name = 'VNet'
    # model_name = "AttentionUnet3D"

    # model_name = 'Unet'

    # 爆显存的模型
    # model_name = 'UNetR'
    # model_name = "TransUnet"

    ModelDict['Unet3D'] = UNet
    ArgsDict['Unet3D'] = {'spatial_dims': 3, 'in_channels': in_channels, 'out_channels': n_classes,
                          'channels': (32, 64, 128, 256, 512), 'strides': (2, 2, 2, 2),
                          'num_res_units': 2, 'bias': True,
                          'dropout': 0.2,
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
