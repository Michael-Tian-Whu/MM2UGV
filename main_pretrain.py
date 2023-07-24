'''
Author: WHURS-THC
Date: 2023-07-23 09:50:58
LastEditTime: 2023-07-24 11:53:40
Description: 
'''
import argparse
from projects.pretrain_ssl import pretrain

def parse_args():
    '''
    description: get args
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpunum",
                        type=int,
                        default=4,
                        help="gpu number")
    parser.add_argument("--k",
                        type=int,
                        default=16,
                        help="cluster number")
    parser.add_argument("--seed",
                        type=int,
                        default=2023,
                        help="random seed")
    parser.add_argument("--model",
                        type=str,
                        default="ResNet50",
                        help="vision model",
                        choices=["ResNet50","ResNet101","ResNet152","Swin-B","Swin-T","Swin-S","ViT-B"])
    parser.add_argument("--epoch",
                        type=int,
                        default=4,
                        help="epoch number")
    parser.add_argument("--subepoch",
                        type=int,
                        default=15,
                        help="subepoch number")
    parser.add_argument("--lr",
                        type=float,
                        default=0.03,
                        help="learning rate")
    parser.add_argument("--batchsize",
                        type=int,
                        default=32,
                        help="batch size")

    args = parser.parse_args()
    return args

def main():
    # get hyperparameters
    args=parse_args()

    gpunum=args.gpunum
    k=args.k
    model=args.model
    epoch=args.epoch
    subepoch=args.subepoch
    lr=args.lr
    batchsize=args.batchsize

    # pretrain
    pretrain(gpunum,k,model,epoch,subepoch,lr,batchsize)

if __name__=="__main__":
    main()