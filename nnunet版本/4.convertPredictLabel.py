import os
from glob import glob


def nii_gz2mhd(path: str):
    import SimpleITK as stk
    path = os.path.abspath(path)
    files = glob(os.path.join(path, "*.nii.gz"))
    idx = 1
    for file in files:
        img = stk.ReadImage(file)
        # .nii.gz
        save_path = 'test_%02d' % idx + ".mhd"
        save_path = os.path.join(path, save_path)
        stk.WriteImage(img, save_path)
        idx += 1

    for file in files:  # 删除nii.gz文件
        os.remove(file)
    pkl_file = glob(os.path.join(os.path.join(path, "*.pkl")))
    for file in pkl_file:
        os.remove(file)


if __name__ == '__main__':
    path = r'./labels'
    nii_gz2mhd(path)
