import bisect
import itertools
import json
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from collections import Counter
__all__ = ["MultiDataSet", "train_set", "test_set", "get_data", "get_data_from_end", "ConvNet", "MLP", "ResAdd",
           "GaussianNoise", "ToTensor", ]

train_set = [
    "20211229_climbing1",
    "20211229_climbing2",
    "20211229_climbing3",
    "20211229_downhill1",
    "20211229_downhill2",
    "20211229_downhill3",
    "20211229_gravel1",
    "20211229_gravel2",
    "20211229_gravel3",
    "20211229_jiyan1",
    "20211229_jiyan2",
    "20211229_jiyan3",
    "20211229_jiyan4",
    "20211229_jiyan5",
    "20211229_jiyan6",
    "20211229_sand1",
    "20211229_sand2",
    "20211229_sand3",
    "20211230_aokeng1",
    "20211230_aokeng2",
    "20211230_aokeng3",
    "20211230_climbing1",
    # "20211230_climbing2",#0
    # "20211230_downhill1",#0
    "20211230_round1",
    "20211230_round2",
    "20211230_round3",
    "20211230_round4",
    "20211230_round5",

]

test_set = [
    "20211229_climbing4",
    "20211229_downhill4",
    "20211229_gravel4",
    "20211229_jiyan7",
    "20211229_jiyan8",
    "20211229_sand4",
    "20211230_aokeng4",
    "20211230_climbing3",
    # "20211230_downhill2",#0
    "20211230_round6",
]


def get_end(s: int, ns: int, table: pd.DataFrame):
    """an option way to get image and sequence aligned"""
    idx = bisect.bisect_left(list(table["time"]), (s, ns))
    dis = 0
    for i in range(idx + 1, table.shape[0]):
        tmp = 0
        tmp += (table["pose_position_x"].iloc[i - 1] - table["pose_position_x"].iloc[i]) ** 2
        tmp += (table["pose_position_y"].iloc[i - 1] - table["pose_position_y"].iloc[i]) ** 2
        dis += tmp ** 0.5
        if dis > 0.83:
            return table[["secs", "nsecs"]].iloc[i]
    return -1, -1


class SingleDataSet(Dataset):
    '''
    单个数据文件
    '''
    def __init__(self, file_dir, **kwargs):
        self.file_dir = file_dir
        self.root=os.path.join(os.getcwd(),"ExtraTerrestrial")

        self.img:pd.DataFrame
        self.stereo:pd.DataFrame
        self.imu:pd.DataFrame
        self.odom:pd.DataFrame

        try:
            # print(os.path.join(self.root, file_dir, "annotation.json"))
            self.annotation = json.load(open(os.path.join(self.root, file_dir, "annotation.json")))    
            self.loc = json.load(open(os.path.join(self.root, file_dir, "loc.json")))
        except FileNotFoundError:
            self.annotation = None
            self.loc = None

        try:
            self.img = pd.read_csv(os.path.join(self.root, file_dir, "img_info.csv"))
            self.img["time"] = self.img.apply(lambda x: (x["secs"], x["nsecs"]), axis=1)
            try:
                self.img["time_end"] = self.img.apply(lambda x: (x["end_s"], x["end_ns"]), axis=1)
            except KeyError:
                self.img["time_end"] = self.img.apply(lambda x: (x["secs"] + 4, x["nsecs"]), axis=1)
            self.stereo = self.img.loc[self.img["image_type"] == "stereo"].copy()
            if self.stereo.empty:
                self.stereo = self.img.loc[self.img["image_type"] == "right"].copy()
            self.stereo["landscape"] = -1
            self.stereo["terrain"] = -1
        except FileNotFoundError:
            self.img = self.binocular = self.stereo = None

        try:
            self.imu = pd.read_csv(os.path.join(self.root, file_dir, "imu.csv"))
            if np.sum(self.imu["frame_id"] == "cv5_link") > 0:
                self.imu = self.imu.loc[self.imu["frame_id"] == "cv5_link"].copy()
            self.imu["time"] = self.imu.apply(lambda x: (x["secs"], x["nsecs"]), axis=1)
            self.imu["angular_x"] = np.cumsum(self.imu["angular_velocity_x"])
            self.imu["angular_y"] = np.cumsum(self.imu["angular_velocity_y"])
            self.imu["angular_z"] = np.cumsum(self.imu["angular_velocity_z"])
        except FileNotFoundError:
            self.imu = None

        try:
            self.odom = pd.read_csv(os.path.join(self.root, file_dir, "odom.csv"))
            self.odom["time"] = self.odom.apply(lambda x: (x["secs"], x["nsecs"]), axis=1)
        except FileNotFoundError:
            self.odom = None

        self.kwargs = kwargs
        print(self.file_dir, self.stereo.shape)

        self.get_item = lambda x, y: 0

    def __getitem__(self, idx: int):
        return self.get_item(idx, self)

    def __len__(self):
        return self.stereo.shape[0]


def terrain_landscape_add(dataset: SingleDataSet):
    '''
    添加地形和地貌的label
    '''
    #地形
    if "downhill" in dataset.file_dir:
        dataset.stereo["terrain"] = 1
    elif "climbing" in dataset.file_dir:
        dataset.stereo["terrain"] = 2
    else:
        dataset.stereo["terrain"] = 0
    dataset.stereo["landscape"] = -1
    #地貌
    if not dataset.annotation:#查看是否有地貌标记
        return
    tmp = [-1, ] * dataset.stereo.shape[0]
    for i in range(dataset.stereo.shape[0]):
        if dataset.annotation["sand"]:
            for j in dataset.annotation["sand"]:
                s, e = j
                if s <= dataset.stereo["seq"].iloc[i] <= e:
                    tmp[i] = 0
        if dataset.annotation["gravel"]:
            for j in dataset.annotation["gravel"]:
                s, e = j
                if s <= dataset.stereo["seq"].iloc[i] <= e:
                    tmp[i] = 1
        if dataset.annotation["rock"]:
            for j in dataset.annotation["rock"]:
                s, e = j
                if s <= dataset.stereo["seq"].iloc[i] <= e:
                    tmp[i] = 2
    dataset.stereo["landscape"] = tmp


class MultiDataSet(Dataset):
    '''
    所有数据文件
    '''
    def __init__(self, dirs, **kwargs):
        super(MultiDataSet, self).__init__()
        self.seq = list(SingleDataSet(d, **kwargs) for d in dirs)
        self.length = []
        self.kwargs = kwargs
        self.label = pd.DataFrame({}, columns=["landscape", "terrain"])

    def get_ready(self, arrive=True, labeled=True):
        for i, v in enumerate(self.seq):
            terrain_landscape_add(v)#添加标记
        self.seq = list(i for i in self.seq if i.stereo.shape[0] > 0)

        connter_1=Counter()
        connter_2=Counter()
        for i, v in enumerate(self.seq):
            if arrive:
                v.stereo = v.stereo.loc[v.stereo["time_end"] != (-1, -1)]
            if labeled:
                v.stereo = v.stereo.loc[v.stereo["landscape"] != -1]
            v.stereo = v.stereo.iloc[8:]
            v.stereo.reset_index(drop=True,inplace=True)#重置索引
            self.label = pd.concat([self.label, v.stereo[["landscape", "terrain"]].copy()], axis=0, ignore_index=True)
            #统计类别
            dict1=v.stereo["landscape"].value_counts().to_dict()
            dict2=v.stereo["terrain"].value_counts().to_dict()
            connter_1+=Counter(dict1)
            connter_2+=Counter(dict2)
        self.length = [0] + list(itertools.accumulate(len(i) for i in self.seq))
        print(f"dataset num:{self.length[-1]}")
        print(f"landscape num:{connter_1}")
        print(f"terrain num:{connter_2}")

    def __len__(self):
        return self.length[-1]

    def __getitem__(self, idx):
        ith = bisect.bisect_right(self.length, idx) - 1
        idx -= self.length[ith]
        return self.seq[ith][idx]

def get_data(start: tuple, end: tuple, col: list, table: pd.DataFrame) -> pd.DataFrame:
    # 给定列，给出指定时间范围内的数据
    start_s, start_ns = start
    end_s, end_ns = end
    row = (table["secs"] == start_s) & (table["nsecs"] >= start_ns)
    row |= (table["secs"] == end_s) & (table["nsecs"] <= end_ns)
    row |= (table["secs"] > start_s) & (table["secs"] < end_s)
    return table[col].loc[row]

class GaussianNoise(object):
    def __init__(self, mu=0, var=0.01):
        self.mu = mu
        self.var = var

    def __call__(self, img, *args, **kwargs):
        noise = torch.randn(img.shape)
        noise *= self.var ** 0.5
        noise += self.mu
        return img + noise
