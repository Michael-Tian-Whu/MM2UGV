import argparse

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--name",default="a",type=str,choices=["a","b"])
    args=parser.parse_args()
    name=args.name
    print(name)
