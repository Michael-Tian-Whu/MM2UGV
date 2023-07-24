import argparse
from projects.test_fewshot import test
def parse_args():
    '''
    description: get args
    '''
    parser = argparse.ArgumentParser()
    # 
    parser.add_argument("--gpunum",
                        type=int,
                        default=4,
                        help="gpu number")
    parser.add_argument("--model",
                        type=str,
                        default="ResNet50",
                        help="vision model",
                        choices=["ResNet50","ResNet101","ResNet152","Swin-B","Swin-T","Swin-S","ViT-B"])
    parser.add_argument("--dir1",
                        type=str,
                        default="./v1_k16_epoch15_4_pre/vision_encoder_14",
                        help="vision encoder dir")
    parser.add_argument("--dir2",
                        type=str,
                        default="./v1_k16_epoch15_4_pre/sensor_encoder_14",
                        help="sensor encoder dir")
    
    args = parser.parse_args()
    return args

def main():
    '''
    description: main
    '''
    args=parse_args()
    gpunum=args.gpunum
    model=args.model
    dir1=args.dir1
    dir2=args.dir2

    # test
    test(gpunum,model,dir1,dir2)


if __name__=="__main__":
    main()