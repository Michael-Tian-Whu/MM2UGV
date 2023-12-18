
import os
import argparse
import numpy as np

import tqdm

import torch
import torch.nn as nn
from torch.utils.data import distributed,DataLoader,Dataset
from torch.utils.tensorboard.writer import SummaryWriter  

import torch.multiprocessing as mp
import torch.distributed as dist_porc
import torch.utils.data.distributed as dist_data
from torch.nn.parallel import DistributedDataParallel as DDP
 

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

n_epoch = 100
batch_size = 64


def main(fun, world_size):
    #tcp通信的ip和端口
    #!不需要在每个进程再设置
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12312'
    #运行进程组 fun()第一个参数默认默认为进程序号
    mp.spawn(fun, args=(world_size,), nprocs=world_size, join=True)


class subDataset(Dataset):
    #初始化，定义数据内容和标签
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label
    #返回数据集大小
    def __len__(self):
        return len(self.Data)
    #得到数据内容和标签
    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.Tensor(self.Label[index])
        return data, label
    


def main_worker(rank, world_size, cpt=None):

    print(os.environ['MASTER_PORT'])
    # 设置随机数种子
    # 实现每个进程的随机数相同，保持进程的同步
    torch.manual_seed(0)

    # 初始化进程组
    dist_porc.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # 分配数据
    data=torch.tensor([[1,2,3],[4,5,6],[7,8,9]]).repeat(3,1)
    label=torch.tensor([0,1,2]).repeat(3)
    dataset=subDataset(data,label)
    sampler = dist_data.DistributedSampler(dataset)#数据分发给该GPU 已经进行了shuffle
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler,pin_memory=True)#sampler自定义采样策略

    #epoch_iter = tqdm.trange(1,unit="epoch") if rank == 0 else range(1)

    torch.cuda.set_device(rank)#低优先级的GPU设置

    for epoch in range(2):
        # 每个GPU每个epoch获得不同的随机数据，否则每次都相同
        sampler.set_epoch(epoch)
        #使所有GPU同步：
        # 进程在此block直到所有进程都进入此函数
        # ！训练过程中进程会自动在all-reduce更新graidents前等待。
        dist_porc.barrier()
        if rank==0:
            print(f"#------epoch:{epoch}------#")
        #训练数据
        train(rank,dataloader)

    dist_porc.destroy_process_group()



def train(rank:int,dataloader):
    for i,data in enumerate(dataloader):
        # 数据在CPU缓存，需要再把数据转到对应GPU
        # 正常情况cuda() default=0 但ddp环境下自动分给rank
        img=data[0].float().cuda()
        label=data[1].float().cuda()

        print(rank,img.device)


if __name__=="__main__":
    assert torch.cuda.is_available(),"pytorch_gpu config failed"

    parser=argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, default="DDP测试")
    arg=parser.parse_args()
    print(arg.desc)

    # gpu数量
    n_gpu=torch.cuda.device_count()
    # gpu属性
    for i in range(n_gpu):
        print(f"gpu {i}")
        print(torch.cuda.get_device_properties(i))
    #运行主函数
    main(main_worker,world_size=n_gpu)