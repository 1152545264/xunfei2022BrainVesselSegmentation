import os
from glob import glob


def nii_gz2mhd(path: str):
    import SimpleITK as stk
    path = os.path.abspath(path)
    files = glob(os.path.join(path, "*.nii.gz"))
    for file in files:
        img = stk.ReadImage(file)
        # .nii.gz
        save_path = file.split(".")[0]  # file[:-7]
        save_path = save_path + ".mhd"
        stk.WriteImage(img, save_path)

    for file in files:  # 删除nii.gz文件
        os.remove(file)


def mhd2nii(path: str):
    import SimpleITK as stk
    path = os.path.abspath(path)
    files = glob(os.path.join(path, "*.mhd"))
    for file in files:
        img = stk.ReadImage(file)
        # .nii.gz
        save_path = file.split(".")[0]
        save_path = save_path + ".nii.gz"
        stk.WriteImage(img, save_path)


if __name__ == '__main__':
    labelsTr = r'./data/labelsTr'
    mhd2nii(labelsTr)
