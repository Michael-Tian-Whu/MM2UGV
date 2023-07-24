'''
Author: WHURS-THC
Date: 2023-07-23 14:02:03
LastEditTime: 2023-07-24 11:33:06
Description: 
'''
'''
Author: WHURS-THC
Date: 2023-07-23 09:50:58
LastEditTime: 2023-07-24 11:07:59
Description: 
'''
import argparse

def parse_args():
    '''
    description: get args
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpunum",type=int,default=4,help="gpu number",metavar="N")

    args = parser.parse_args()
    return args

def main():

    args=parse_args()
    print(args)

if __name__=="__main__":
    main()