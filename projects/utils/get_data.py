from .toolbox import SingleDataSet,GaussianNoise
import numpy as np
from PIL import Image, ImageOps
import os
import torch
from .toolbox import get_data
from scipy import signal
from torchvision import transforms as T
import bisect
from torch.utils.data import DataLoader, Dataset
import tqdm
from scipy.spatial.transform import Rotation

__all__ = ["get_item", "get_landscape", "get_terrain", "get_visual_cluster", "get_sensor_cluster", "get_image",
           "get_sensor", "train_trans", "test_trans", "image_seq", "sensor_seq", "get_loc", "get_trajectory",
           "get_file_dir", "get_state_seq", "get_raw_imu", "get_raw_odom", "get_idx", "get_yaw", "get_throttle",
           "get_velocity_x", "get_image_name", "get_imu_acc_z"]
train_trans = T.Compose([
    T.RandomApply([ # 图像颜色抖动
                T.ColorJitter(brightness = (0.8, 1.2),contrast = (0.8, 1.2), saturation = (0.8, 1.2), hue = (-0.1, 0.1)) 
            ], p=0.8),
    T.RandomGrayscale(p=0.2),# 灰度化
    T.RandomApply([T.GaussianBlur( kernel_size=(3, 3))], p=0.5),#高斯模糊
    T.Resize((224,224)),
    T.ToTensor(),    # ndarray->tensor (0,1)
    T.Normalize(mean=(0.5, 0.5, 0.5),# 归一化 直方图标准化后U(0,1) 
                         std=(1 / 12 ** 0.5, 1 / 12 ** 0.5, 1 / 12 ** 0.5,)),
])

test_trans = T.Compose([
    T.Resize(size=(224, 224)),
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(1 / 12 ** 0.5, 1 / 12 ** 0.5, 1 / 12 ** 0.5,)),
])

def get_item(idx: int, obj: SingleDataSet):
    if not obj.kwargs.get("attribute") or not obj.kwargs.get("util"):
        return 0
    res = ()
    for attr in obj.kwargs["attribute"]:
        func = obj.kwargs["util"].get(attr, lambda x, y, **kwargs: 0)
        res += (func(idx, obj, **obj.kwargs.get("function_helper", dict()).get(attr, dict())),)
    return res


def get_landscape(idx: int, obj: SingleDataSet, **kwargs):
    return obj.stereo["landscape"].iloc[idx]


def get_terrain(idx: int, obj: SingleDataSet, **kwargs):
    return obj.stereo["terrain"].iloc[idx]


def get_visual_cluster(idx: int, obj: SingleDataSet, **kwargs):
    return obj.stereo["v_cluster"].iloc[idx]


def get_sensor_cluster(idx: int, obj: SingleDataSet, **kwargs):
    return obj.stereo["s_cluster"].iloc[idx]


def get_image(idx: int, obj: SingleDataSet, **kwargs):
    trans = kwargs.get("trans", lambda x: x)
    up = kwargs.get("up", True)
    file_name = obj.stereo["file_name"].iloc[idx].replace("stereo", "floor")
    img = np.array(Image.open(os.path.join(obj.root,obj.file_dir, file_name)))
    h, w, c = img.shape#400,800,3
    if up:#上半部分 0-200,500-800
        res = img[:h // 2, w // 8 * 3:w // 8 * 5, :]
    else:#下半部分
        res = img[h // 2:, w // 8 * 3:w // 8 * 5, :]
    # res = ImageOps.equalize(Image.fromarray(res))#
    res = Image.fromarray(res)
    return res#,trans(res)


def get_image_name(idx: int, obj: SingleDataSet, **kwargs):
    file_name = obj.stereo["file_name"].iloc[idx].replace("stereo", "floor")
    return file_name


def get_sensor(idx: int, obj: SingleDataSet, **kwargs):
    '''
    :返回传感器数据(1,length,11)
    '''
    trans = kwargs.get("trans", lambda x: x)
    length = kwargs.get("input_length", 10)
    split = kwargs.get("split", "arrive")
    s = obj.stereo["time"].iloc[idx]
    if split == "arrive":
        e = obj.stereo["time_end"].iloc[idx]
    elif type(split) == int:
        e = obj.stereo["time_end"].iloc[idx + split]
    else:
        e = obj.stereo["time_end"].iloc[idx]

    # imu线加速度 (6) 150hz
    # y轴受人为控制，不考虑
    avg = torch.nn.AdaptiveAvgPool1d(output_size=length)
    arr = get_data(s, e, col=["linear_acceleration_x", "linear_acceleration_z"], table=obj.imu)
    f, n = arr.shape
    acc = arr.to_numpy()
    acc[:, -1] -= np.mean(acc[:, -1])
    res_acc = np.zeros((length, 3 * n))
    for i in range(n):
        col = acc[:, i]
        #短时傅里叶变换 
        #fs (fs/2) 采集频率 
        #nperseg(N/nperseg/overlap) 分割时间点
        #zxx (fs/2,N/nperseg/overlap)
        f, t, zxx = signal.stft(col, fs=150, nperseg=150)
        zxx = np.absolute(zxx)[:length, :]#10hz以上的视为噪声
        res_acc[:, 2 * i] = np.mean(zxx, axis=1)  # x, z 加速度频域
        res_acc[:, 2 * i + 1] = np.var(zxx, axis=1)  # x, z 加速度频域
        res_acc[:, 2 * i + 2] = avg(torch.tensor(col).view((1, 1, -1))).numpy().squeeze()  # x, z轴加速度时域

    # imu角速度 (4) 150hz
    # z轴受人为控制，不考虑
    arr = get_data(s, e, col=['angular_velocity_x', 'angular_velocity_y', "angular_x", "angular_y"], table=obj.imu)
    f, n = arr.shape
    angular = arr.to_numpy()
    res_angular = np.zeros((length, n))
    for i in range(n):
        col = angular[:, i]
        res_angular[:, i] = avg(torch.tensor(col).view((1, 1, -1))).numpy().squeeze()

    # odom绝对速度 (2) 40hz
    if obj.odom is not None:
        arr = get_data(s, e, col=["pose_position_x", "pose_position_y"], table=obj.odom)
        arr = arr.to_numpy()
        arr = arr[1:, :] - arr[:-1, :]
        vel = np.sqrt(np.sum(arr * arr, axis=1)) * 40
        vel = avg(torch.tensor(vel).view((1, 1, -1))).numpy().reshape((-1, 1))
    else:
        vel = np.zeros((length, 1))

    # cat all sensor
    # print(res_angular.shape, res_acc.shape, vel.shape)
    sensor = np.concatenate((res_acc, res_angular, vel), axis=1)
    a, b = sensor.shape
    return trans(sensor.reshape((1, a, b)))



def get_imu_acc_z(idx: int, obj: SingleDataSet, **kwargs):
    t1 = obj.stereo["time"].iloc[idx]
    t2 = (t1[0] + 4, t1[0])
    idx_ = (obj.imu["time"] >= t1) & (obj.imu["time"] <= t2)
    return obj.imu["linear_acceleration_z"].loc[idx_]


