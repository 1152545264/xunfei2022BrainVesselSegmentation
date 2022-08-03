# -*-coding:utf-8-*-
import os
from monai.engines import EnsembleEvaluator
from monai.handlers import ValidationHandler

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
from monai.engines import EnsembleEvaluator

from main_version2 import BrainVessel

from TransUnet.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from Config import Config

Config.PredSegDir = os.path.join(Config.data_dir, 'PredSeg', 'model_ensemble', "labels")


def get_predTransformer():
    pred_transform = Compose([
        LoadImaged(keys=['image']),
        EnsureChannelFirstd(keys=['image']),

        # Orientationd(keys=['image', 'label'], axcodes=cfg.AxCode),
        # Spacingd(keys=['image', 'label'], pixdim=cfg.Spacing),

        NormalizeIntensityd(keys=['image'], ),
        EnsureTyped(keys=['image']),
    ])
    return pred_transform


def getTrainer(cfg=Config()):
    trainer = pl.Trainer(
        progress_bar_refresh_rate=10,
        max_epochs=cfg.max_epoch,
        min_epochs=cfg.min_epoch,
        gpus=1,
        auto_scale_batch_size=True,  # 这个参数对模型训练速度很关键
        # accumulate_grad_batches=4, # 慎用，否则log里面显示出来的train_mean_loss会有很大的问题
        precision=16,
        log_every_n_steps=5000,
        num_sanity_val_steps=0,
        gradient_clip_val=1e3,
        gradient_clip_algorithm='norm',
    )
    return trainer


def getModel(model_name):
    cfg = Config()
    ckptFiles = glob(os.path.join(f'./logs/{cfg.model_name}', "*.ckpt"))
    bestFile = ""
    bestScore = 0.0
    for file in ckptFiles:
        score = file.split("=")[-1][:-5]
        score = float(score)
        if bestScore < score:
            bestFile, bestScore = file, score

    # model = cfg.ModelDict[model_name]
    # args = cfg.ArgsDict[model_name]
    model = BrainVessel.load_from_checkpoint(bestFile)  # fixme，验证是否正确
    return model


def getSameModelMultiCkpt(model_name):  # 返回同一个模保存下来的多个ckp权值
    ckptFiles = glob(os.path.join(f'./logs/{model_name}', "*.ckpt"))
    return ckptFiles


def getModelBestScoreCkpt(model_name):
    ckptFiles = glob(os.path.join(f'./logs/{model_name}', "*.ckpt"))
    bestFile = ""
    bestScore = 0.0
    for info in ckptFiles:
        score = info.split("=")[-1][:-5]
        score = float(score)
        if bestScore < score:
            bestFile, bestScore = info, score
    return [bestFile]


def get_predict_files():
    predict_dir = os.path.join(Config.data_dir, r'Testing data\Test-MRA')
    predict_files = glob(os.path.join(predict_dir, "*.mhd"))
    tmp = []
    for file in predict_files:
        file_dict = {'image': file}
        tmp.append(file_dict)
    return tmp


def save_pred(meta_dict, y_pred):
    for k, v in meta_dict.items():
        if isinstance(v, torch.Tensor):
            meta_dict[k] = v.detach().cpu()
    saver = NiftiSaver(output_dir=Config.PredSegDir, mode="nearest", print_log=True,
                       separate_folder=False, output_postfix="")  # 禁止为单个文件新建目录
    saver.save_batch(y_pred, meta_dict)  # fixme 检查此处用法是否正确


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


def ensemblePredict(model_names: [str]):
    cfg = Config()
    predict_files = get_predict_files()
    pred_transform = get_predTransformer()
    trainer = getTrainer()
    assert isinstance(trainer, pl.Trainer)

    for (_, pred_file) in enumerate(predict_files):
        image_preds = []
        meta_info = None

        for m_name in model_names:
            # ckptFiles = getSameModelMultiCkpt(m_name)
            ckptFiles = getModelBestScoreCkpt(m_name)

            assert isinstance(ckptFiles, list)
            if len(ckptFiles) == 0:
                continue
            for ckptFile in ckptFiles:
                model = BrainVessel.load_from_checkpoint(ckptFile)
                model.eval()

                single_set = Dataset([pred_file], transform=pred_transform)
                single_loader = DataLoader(single_set, batch_size=1)

                pred_info = trainer.predict(model, single_loader)
                for (y_pred, meta_dict) in pred_info:
                    x_shape = y_pred.shape[2:]
                    if x_shape[0] != cfg.Final_shape[0]:
                        y_pred = Spacing(cfg.OrgSpacing, mode='bilinear')(y_pred)
                    image_preds.append(y_pred)
                    space_info = meta_dict['spatial_shape'][0]
                    if meta_info is None and space_info[0] == cfg.Final_shape[0]:
                        meta_info = meta_dict

        if len(image_preds) == 0:
            exit(0)

        if not cfg.UseSigmoid:
            post_pred = Compose([
                Activations(softmax=True),
                AsDiscrete(argmax=True),
            ])
        else:
            post_pred = Compose([
                EnsureType(),
                Activations(sigmoid=True),
                AsDiscrete(threshold=0.5),
            ])
        tmp = []
        for it in image_preds:
            x = post_pred(decollate_batch(it))
            tmp.append(torch.stack(x, dim=0))
        image_preds = tmp

        # ensemble = MeanEnsemble(num_classes=cfg.n_classes)
        ensemble = VoteEnsemble()
        image_pred = ensemble(image_preds)

        save_pred(meta_info, image_pred)


if __name__ == '__main__':
    model_name = ["Unet3D"]

    ensemblePredict(model_name)

    nii_gz2mhd()
