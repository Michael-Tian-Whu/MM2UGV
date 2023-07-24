import os
from math import cos, pi
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models

from utils.toolbox import MultiDataSet, test_set
from utils.get_data import test_trans
from utils.get_data import get_item, get_sensor, get_image, get_landscape, get_terrain, get_visual_cluster, \
    get_sensor_cluster
from utils.vit import ViT
from utils.metrics import CosineSimTest

from time import time

def test(gpunum=4,model="ResNet50",dir1="./v1_k16_epoch15_4_pre/vision_encoder_14",dir2="./v1_k16_epoch15_4_pre/sensor_encoder_14"):
    '''
    description: test
    '''

    # distribute GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    devices = [i for i in range(gpunum)]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in devices])

    # test set
    num_test = 100
    batch_test = 15

    kwarg = {"attribute": ("sensor", "vision", "landscape", "terrain", "v_cluster", "s_cluster"),
            "util": {"sensor": get_sensor, "vision": get_image, "landscape": get_landscape,
                    "terrain": get_terrain, "v_cluster": get_visual_cluster, "s_cluster": get_sensor_cluster},
            "function_helper": {"vision": {"trans": test_trans}}}
    test = MultiDataSet(test_set, **kwarg)

    for i in range(len(test.seq)):
        test.seq[i].get_item = get_item
        test.seq[i].stereo["v_cluster"] = -1
        test.seq[i].stereo["s_cluster"] = -1
        test.seq[i].stereo.reset_index(drop=True,inplace=True)#重置索引
    test.get_ready()

    #vision encoder init
    if model=="ResNet50":

        vision_encoder = models.resnet50()
        vision_encoder.fc = nn.Linear(in_features=2048, out_features=128)
    elif model=="ResNet101":

        vision_encoder = models.resnet101()
        vision_encoder.fc = nn.Linear(in_features=2048, out_features=128)
    elif model=="ResNet152":

        vision_encoder = models.resnet152()
        vision_encoder.fc = nn.Linear(in_features=2048, out_features=128)
    elif model=="ViT_B":  

        vision_encoder=models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        vision_encoder.heads = nn.Linear(in_features=768, out_features=128)
    elif model=="Swin_T":

        vision_encoder = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        vision_encoder.head = nn.Linear(in_features=768, out_features=128)
    elif model=="Swin_S":

        vision_encoder = models.swin_s(weights=models.Swin_S_Weights.DEFAULT)
        vision_encoder.head = nn.Linear(in_features=768, out_features=128)
    elif model=="Swin_B":

        vision_encoder = models.swin_b(weights=models.Swin_B_Weights.DEFAULT)
        vision_encoder.head = nn.Linear(in_features=1024, out_features=128)
    
    # sensor encoder init
    sensor_encoder = ViT(image_size=(10, 11),patch_size=(10, 1),num_classes=128,
        dim=16,depth=6,heads=16,mlp_dim=256,
        dropout=0.1,emb_dropout=0.1,channels=1
    )
    vision_encoder=nn.DataParallel(vision_encoder,device_ids=devices).cuda(devices[0])
    sensor_encoder=nn.DataParallel(sensor_encoder,device_ids=devices).cuda(devices[0])

    # load pretrain model

    vision_encoder.load_state_dict(
        torch.load(dir1,map_location=f"cuda:{devices[0]}"))
    sensor_encoder.load_state_dict(
        torch.load(dir2,map_location=f"cuda:{devices[0]}"))

    vision_encoder = vision_encoder.module.eval()
    sensor_encoder = sensor_encoder.module.eval()

    stime=time()
    # fewshot test
    cri = CosineSimTest(vision_model=vision_encoder, sensor_model=sensor_encoder, test_data=test,
                            n_test=num_test, batch_size=batch_test, devices=devices)
    res = cri.eval()
    etime=time()
    s_landscape_acc, _, s_terrain_acc, _, \
    v_landscape_acc, _, v_terrain_acc, _, \
    f_landscape_acc, _, f_terrain_acc, _=res 

    print(f"inference time {etime-stime}")
    print(f'vision prec task1:{v_landscape_acc:.3f} task2:{v_terrain_acc:.3f}')
    print(f'sensor prec task1:{s_landscape_acc:.3f} task2:{s_terrain_acc:.3f}')
    print(f'fusion prec task1:{f_landscape_acc:.3f} task2:{f_terrain_acc:.3f}')

