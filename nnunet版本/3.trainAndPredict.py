from genericpath import isdir
from importlib.resources import path
import os
import shutil
from sys import prefix
from glob import glob

from numpy import full, full_like

preprocessFolder = "nnUNet_preprocessed"
rawFolder = "nnUNet_raw"
trainedModelsFolder = "nnUNet_trained_models"
necessaryFolders = [preprocessFolder, rawFolder, trainedModelsFolder]

curPath = os.path.abspath("./")
task_name = "Task503"
data_prefix = "KDXF"
src_dir = "./data"

root_path = os.path.join(curPath, "DATASET")
mp = {
    "nnUNet_raw_data_base": os.path.join(root_path, rawFolder),
    "nnUNet_preprocessed": os.path.join(root_path, preprocessFolder),
    "RESULTS_FOLDER": os.path.join(root_path, trainedModelsFolder)
}

raw_data_folder = os.path.join(
    root_path, rawFolder, "nnUNet_raw_data")


def createFolerders():
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    for folder in necessaryFolders:
        full_path = os.path.join(root_path, folder)
        if not os.path.exists(full_path):
            os.mkdir(full_path)

    if not os.path.exists(raw_data_folder):
        os.mkdir(raw_data_folder)

    fullTaskName = task_name + "_" + data_prefix
    fullPath = os.path.join(raw_data_folder, fullTaskName)
    if not os.path.exists(fullPath):
        os.mkdir(fullPath)


def generateTask(path=os.path.join(src_dir, "Task503_KDXF")):
    orders = f'nnUNet_convert_decathlon_task -i {path}'
    os.system(orders)


def moveFolder():
    src = src_dir
    dst = os.path.join(
        raw_data_folder, task_name + "_" + data_prefix)

    folders = ["imagesTr", "imagesTs", "labelsTr", "dataset.json"]
    for folder in folders:
        src_fp = os.path.join(src, folder)
        dst_fp = os.path.join(dst, folder)
        if not os.path.exists(src_fp):  # 如果源文件夹或者源文件不存在，直接返回
            continue

        # 已经存在，则需要先删除对应的文件再进行拷贝
        if os.path.exists(dst_fp):
            if os.path.isfile(dst_fp):
                os.remove(dst_fp)
            elif os.path.isdir(dst):
                shutil.rmtree(dst_fp)

        if os.path.isfile(src_fp):
            shutil.copy(src_fp, dst_fp)
        elif os.path.isdir(src_fp):
            shutil.copytree(src_fp, dst_fp)


def addEnv(mp: dict):
    for (k, v) in mp.items():
        # os.system(f"export {k}={v}")
        os.environ[k] = v


def renameFiles(dst: str = os.path.join(
    raw_data_folder, task_name + "_" + data_prefix)):
    folders = ["imagesTr", "imagesTs"]  # label不需要重命名
    for folder in folders:
        full_path = os.path.join(dst, folder)
        files = glob(os.path.join(full_path, "*.nii.gz"))
        for file in files:
            name = file.split('/')[-1]
            name = name.split('.')[0] + '_0000' + '.nii.gz'
            n_name = os.path.join(full_path, name)
            os.rename(file, n_name)


def train():
    task_idx = task_name[4:]
    checkorder = f"nnUNet_plan_and_preprocess -t {task_idx} --verify_dataset_integrity"
    os.system(checkorder)
    fullTaskName = task_name + "_" + data_prefix
    # train_order = f"nnUNet_train 3d_fullres nnUNetTrainerV2 {fullTaskName} all"
    train_order = f"nnUNet_train 3d_fullres nnUNetTrainerV2 {fullTaskName} 5"
    os.system(train_order)


def predict():
    pred_label = os.path.abspath(r'./labels')
    if not os.path.exists(pred_label):
        os.mkdir(pred_label)

    order = f'nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task503_KDXF/imagesTs/ \
                            -o {pred_label} \
                            -t {task_name[4:]} \
                            -m 3d_fullres \
                            -f 5 '
    os.system(order)


if __name__ == "__main__":
    # train
    # createFolerders()
    # moveFolder()
    # generateTask()
    # addEnv(mp)
    # renameFiles()
    # train()

    # predict
    addEnv(mp)
    predict()
