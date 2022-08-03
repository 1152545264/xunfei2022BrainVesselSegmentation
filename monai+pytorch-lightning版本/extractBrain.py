import os
from glob import glob
import SimpleITK as stk
import time

''' 
    FSL安装：
        先将清华镜像源添加到Ubuntu的软件源，参考：https://zhuanlan.zhihu.com/p/479322687
        再按照下述博客的方法(官网通过python脚本安装的方法，开梯子都会失败)进行安装，https://blog.csdn.net/jiangjiang_jian/article/details/80698029
        执行命令：source /etc/bash.bashrc
        
    
    将原始数据集解压后得到的文件夹改名为TrainingData和TestingData(因为FSL工具包不支持文件路径中有空格)
    1、运行本脚本进行脑组织提取
    2、提取完成之后，需要新建文件夹TestingDataRemoveBone/Test-MRA和TrainingDataRemoveBone/Tran-MRA,
    3、将TestingData/Test-MRA下的*.nii.gz文件拷贝到TestingDataRemoveBone/Test-MRA下
    4、将TrainingData/Tran-MRA下的*_brain.nii.gz文件全部拷贝到TrainingDataRemoveBone/Tran-MRA下，
        并将TrainingData/Tran-MRA-labels文件夹拷贝到TrainingDataRemoveBone下
    5、运行main_version.py脚本，进行模型训练和预测
 '''

train_path = r"/DATA_DISK/Caiyimin/Dataset/Small3D/KeDaXunFei2022BrainVessel/TrainingData/Tran-MRA"
test_path = r"/DATA_DISK/Caiyimin/Dataset/Small3D/KeDaXunFei2022BrainVessel/TestingData/Test-MRA"

base_path = r"/DATA_DISK/Caiyimin/Dataset/Small3D/KeDaXunFei2022BrainVessel/"


def mhd2nii(path=train_path):
    # import SimpleITK as stk
    files = glob(os.path.join(path, "*.mhd"))
    for file in files:
        print(f"Convert {file} to nii.gz")
        img = stk.ReadImage(file)
        # .nii.gz
        save_path = file.split(".")[0]
        save_path = save_path + ".nii.gz"
        stk.WriteImage(img, save_path)


# extractBrain函数需要运行在Linux系统下，且该系统已经安装好了FSL工具
def extractBrain(path=train_path):
    files = glob(os.path.join(path, "*.nii.gz"))
    thresh = 0.2
    for file in files:
        outfile = file.split(".")[0] + "_brain"
        order = f"bet2 {file} {outfile} -f {thresh}"
        # order = "bet2 " + file + " outfile" + " -f" + str(thresh)
        print(order)
        os.system(order)

        # org_name = file.split(".")[0] + "Org.nii.gz"
        # os.system(fr"mv {file}  {org_name}")
        #
        # brain_name = outfile.split("_")[0] + ".nii.gz"
        # os.system(fr'mv {outfile} {brain_name}')

        time.sleep(5)


if __name__ == "__main__":
    mhd2nii()
    mhd2nii(test_path)

    extractBrain(train_path)
    extractBrain(test_path)
