import os
import copy
from math import cos, pi
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models
from torch.utils import tensorboard
from torch.utils.data import DataLoader

from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score

from utils.toolbox import MultiDataSet, train_set, test_set
from utils.get_data import train_trans, test_trans
from utils.get_data import get_item, get_sensor, get_image, get_landscape, get_terrain, get_visual_cluster, \
    get_sensor_cluster
from utils.vit import ViT

from tqdm import trange,tqdm
from prefetch_generator import BackgroundGenerator
from time import time


class Logger(tensorboard.SummaryWriter):
    '''
    data record
    e.g. tensorboard --logdir=./xdc/exp6 --port 1234
    '''
    def __init__(self, log_dir):
        super(Logger,self).__init__(log_dir)
        self.counter = Counter()

    def log_scalar(self, tag, value):
        self.add_scalar(tag, value, self.counter[tag])
        self.counter[tag] += 1

class DataLoaderX(DataLoader):
    '''
    Transfer the required data to GPU before the next batch, 
    no need to wait, improve GPU utilization.
    '''
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def projector(out_feature):
    '''
    classification head
    '''
    return nn.Sequential(
        nn.GELU(),
        nn.BatchNorm1d(num_features=128),
        nn.Linear(in_features=128, out_features=128),
        nn.GELU(),
        nn.BatchNorm1d(num_features=128),
        nn.Linear(in_features=128, out_features=out_feature),
        )

def init_weight(net:nn.Module):
    '''
    init the weight of classification head
    '''
    for name, param in net.named_parameters():
        if int(name[0])>2:#输出层更新
            if 'weight' in name:
                nn.init.uniform_(param) 
            if 'bias' in name: 
                nn.init.constant_(param, val=0) 

def warmup_cosine(optimizer, current_epoch,max_epoch,lr_min=0,lr_max=0.1,warmup=True):
    '''
    warmup+cosine lr
    '''
    warmup_epoch = 5 if warmup else 0
    if current_epoch < warmup_epoch:
        lr =  (lr_max-lr_min)  * current_epoch / warmup_epoch+lr_min
    else:
        lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def pretrain(gpunum=4,k=16,seed=2023,model="ResNet50",epoch=4,subepoch=15,lr=0.03,batchsize=32):
    '''
    pretrain the vision and sensor encoder
    '''
    # Distribute GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    devices = [i for i in range(gpunum)]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in devices])

    # lr
    lr_min=1e-4
    lr_max=lr
    lr_cons=1e-3

    # log
    pre_model="pre"#scratch/pre
    exp_no=f"exp0"
    writer=Logger(f"./projects/tensorboard/{exp_no}")

    # dataset
    kwarg = {"attribute": ("sensor", "vision", "landscape", "terrain", "v_cluster", "s_cluster"),
            "util": {"sensor": get_sensor, "vision": get_image, "landscape": get_landscape,
                    "terrain": get_terrain, "v_cluster": get_visual_cluster, "s_cluster": get_sensor_cluster},
            "function_helper": {"vision": {"trans": train_trans}}}
    train = MultiDataSet(train_set, **kwarg)


    # init cluster pseudo label
    for i in range(len(train.seq)):
        train.seq[i].get_item = get_item
        train.seq[i].stereo["v_cluster"] = -1
        train.seq[i].stereo["s_cluster"] = -1
        train.seq[i].stereo.reset_index(drop=True,inplace=True)#reset index

    # get ground truth
    train.get_ready()

    # PCA dimension reduction & whitening
    reduction_s = PCA(n_components=k, whiten=True)
    reduction_v = PCA(n_components=k, whiten=True)
    # K-Means cluster
    cluster_s = KMeans(n_clusters=k,n_init="auto")
    cluster_v = KMeans(n_clusters=k,n_init="auto")
    # # l2norm
    # norm2_s= Normalizer(norm='l2')
    # norm2_v= Normalizer(norm='l2')

    #vision encoder init
    if model=="ResNet50":
        pre_model="scratch"
        vision_encoder = models.resnet50()
        vision_encoder.fc = nn.Linear(in_features=2048, out_features=128)
    elif model=="ResNet101":
        pre_model="scratch"
        vision_encoder = models.resnet101()
        vision_encoder.fc = nn.Linear(in_features=2048, out_features=128)
    elif model=="ResNet152":
        pre_model="scratch"
        vision_encoder = models.resnet152()
        vision_encoder.fc = nn.Linear(in_features=2048, out_features=128)
    elif model=="ViT_B":  
        pre_model="pre" 
        vision_encoder=models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        vision_encoder.heads = nn.Linear(in_features=768, out_features=128)
    elif model=="Swin_T":
        pre_model="pre" 
        vision_encoder = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        vision_encoder.head = nn.Linear(in_features=768, out_features=128)
    elif model=="Swin_S":
        pre_model="pre" 
        vision_encoder = models.swin_s(weights=models.Swin_S_Weights.DEFAULT)
        vision_encoder.head = nn.Linear(in_features=768, out_features=128)
    elif model=="Swin_B":
        pre_model="pre" 
        vision_encoder = models.swin_b(weights=models.Swin_B_Weights.DEFAULT)
        vision_encoder.head = nn.Linear(in_features=1024, out_features=128)
        
    # 
    version=f"v1_k{k}_epoch{epoch}_{subepoch}_{pre_model}_{model}"

    vision_classifier = projector(out_feature=k,)

    vision_encoder = nn.DataParallel(vision_encoder, device_ids=devices).cuda(devices[0])
    vision_classifier = nn.DataParallel(vision_classifier, device_ids=devices).cuda(devices[0])

    # sensor encoder init
    sensor_encoder = ViT(image_size=(10, 11),patch_size=(10, 1),num_classes=128,
        dim=16,depth=6,heads=16,mlp_dim=256,
        dropout=0.1,emb_dropout=0.1,channels=1
    )
    sensor_classifier = projector(out_feature=k)

    sensor_encoder = nn.DataParallel(sensor_encoder, device_ids=devices).cuda(devices[0])
    sensor_classifier = nn.DataParallel(sensor_classifier, device_ids=devices).cuda(devices[0])


    # encoder trainer init
    sensor_encoder_trainer = torch.optim.AdamW(sensor_encoder.parameters(), lr=lr_min)
    vision_encoder_trainer = torch.optim.AdamW(vision_encoder.parameters(), lr=lr_min)
    # classifier trainer init
    sensor_classifier_trainer = torch.optim.Adam(sensor_classifier.parameters(), lr=lr_cons)
    vision_classifier_trainer = torch.optim.Adam(vision_classifier.parameters(), lr=lr_cons)
    loss = nn.CrossEntropyLoss().cuda(devices[0])

    # seed
    torch.manual_seed(seed)

    # dataloader
    cluster_iter = DataLoaderX(dataset=train, batchsize=batchsize, num_workers=os.cpu_count(), shuffle=False)

    # train loop
    sc_last:np.ndarray
    vc_last:np.ndarray
    epoch_iter=trange(epoch,unit="epoch",ncols=100,leave=True)
    for i in epoch_iter:
        print("")

        #step1 cluster pseudo label

        vision_encoder = vision_encoder.float().eval()
        sensor_encoder = sensor_encoder.float().eval()

        sc, vc = torch.zeros((0, 128)), torch.zeros((0, 128))

        # get feature
        for s, v, a, b, v_pseudo_label, s_pseudo_label in cluster_iter:
            s = s.float()#(B,3,224,224)
            v = v.float()#(B,1,10,11)
            tmp_s = sensor_encoder(s.cuda(devices[0])).cpu()#(128)
            tmp_v = vision_encoder(v.cuda(devices[0])).cpu()
            sc = torch.cat((sc, tmp_s.detach()))#concat 
            vc = torch.cat((vc, tmp_v.detach()))
        sc, vc = sc.detach().numpy(), vc.detach().numpy()
    
        #PCA 
        stime=time()
        reduction_s.fit(sc)#(N,128)->(N,32)
        reduction_v.fit(vc)

        #K-Means
        sc = cluster_s.fit_predict(reduction_s.transform(sc))#(N,32)->(N,1)分配类别
        vc = cluster_v.fit_predict(reduction_v.transform(vc))
        if i>0:
            sensor_cluster_nmi = normalized_mutual_info_score(sc_last, sc)
            vision_cluster_nmi = normalized_mutual_info_score(vc_last, vc)
            writer.log_scalar(f"{version}/sensor_cluster_NMI", sensor_cluster_nmi)
            writer.log_scalar(f"{version}/vision_cluster_NMI", vision_cluster_nmi)
        sc_last=sc
        vc_last=vc

        etime=time()
        print(f"cluster time:{etime-stime:.2f}s")
        print(f"image cluster:{Counter(sc)}\nsensor cluster:{Counter(vc)}")
        ith = 0

        # get pseudo label
        for j in range(len(train.seq)):
            for k in range(len(train.seq[j])):
                train.seq[j].stereo.loc[k,"v_cluster"]= int(vc[ith])
                train.seq[j].stereo.loc[k,"s_cluster"]= int(sc[ith])
                ith += 1

        #step2. model update

        #reset lr
        warmup_cosine(sensor_encoder_trainer, i,epoch,lr_min,lr_max,warmup=True)
        warmup_cosine(vision_encoder_trainer, i,epoch,lr_min,lr_max,warmup=True)

        #reset weight of classification head
        init_weight(vision_classifier.module)
        vision_encoder = vision_encoder.float().train()
        vision_classifier = vision_classifier.float().train()
        init_weight(sensor_classifier.module)
        sensor_encoder = sensor_encoder.float().train()
        sensor_classifier = sensor_classifier.float().train()

        # dataloader
        train_iter = DataLoaderX(dataset=train, batchsize=batchsize, num_workers=os.cpu_count(), drop_last=True,shuffle=True)
        for j in range(subepoch):
            print(f"epoch:{i}  subepoch:{j}")
            sensor_pred, vision_pred = torch.zeros((0,)), torch.zeros((0,))
            landscape, terrain = torch.zeros((0,)), torch.zeros((0,))
            for n,(s, v, a, b, v_pseudo_label, s_pseudo_label) in enumerate(train_iter):
                s = s.float().cuda(devices[0])
                v = v.float().cuda(devices[0])
                # feature
                feature_s = sensor_encoder(s)
                feature_v = vision_encoder(v)
                # classification head
                logit_s = sensor_classifier(feature_s)
                logit_v = vision_classifier(feature_v)
                v_pseudo_label = v_pseudo_label.cuda(devices[0])
                s_pseudo_label = s_pseudo_label.cuda(devices[0])
                # loss
                l = 0
                l += loss(logit_s, v_pseudo_label) + loss(logit_v, s_pseudo_label)
                sensor_encoder_trainer.zero_grad()
                sensor_classifier_trainer.zero_grad()
                vision_encoder_trainer.zero_grad()
                vision_classifier_trainer.zero_grad()
                l.backward()
                sensor_encoder_trainer.step()
                sensor_classifier_trainer.step()
                vision_encoder_trainer.step()
                vision_classifier_trainer.step()
                tmp_s = torch.argmax(logit_s.cpu().detach(), dim=-1)
                tmp_v = torch.argmax(logit_v.cpu().detach(), dim=-1)
                sensor_pred = torch.cat((sensor_pred, tmp_s))
                vision_pred = torch.cat((vision_pred, tmp_v))
                landscape = torch.cat((landscape, a))
                terrain = torch.cat((terrain, b))
                if n%10==0:
                    print(l.data)
                    writer.log_scalar(f"{version}/loss", l.data)

        # NMI criterion
        vision_pred = vision_pred.detach().int().tolist()
        sensor_pred = sensor_pred.detach().int().tolist()
        landscape = landscape.detach().int().tolist()
        terrain = terrain.detach().int().tolist()
        # NMI
        vision_landscape_nmi = normalized_mutual_info_score(vision_pred, landscape)
        vision_terrain_nmi = normalized_mutual_info_score(vision_pred, terrain)
        sensor_landscape_nmi = normalized_mutual_info_score(sensor_pred, landscape)
        sensor_terrain_nmi = normalized_mutual_info_score(sensor_pred, terrain)
        writer.log_scalar(f"{version}/vision_landscape_NMI", vision_landscape_nmi)
        writer.log_scalar(f"{version}/vision_terrain_NMI", vision_terrain_nmi)
        writer.log_scalar(f"{version}/sensor_landscape_NMI", sensor_landscape_nmi)
        writer.log_scalar(f"{version}/sensor_terrain_NMI", sensor_terrain_nmi)
        
        # save model
        d = f"./projects/checkpoint/{version}"
        try:
            torch.save(vision_encoder.state_dict(), f"{d}/vision_encoder_{i}")
            torch.save(sensor_encoder.state_dict(), f"{d}/sensor_encoder_{i}")
        except Exception:
            os.makedirs(d)
            torch.save(vision_encoder.state_dict(), f"{d}/vision_encoder_{i}")
            torch.save(sensor_encoder.state_dict(), f"{d}/sensor_encoder_{i}")
    # save cluster result
    data={"sensor":sc,"vision":vc}
    pd.DataFrame(data).to_csv(f"{d}/cluster.csv")

