import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data", help="path to training and testing data", type=str)
parser.add_argument("--dimensions", help="dimension of word vector", type=int)
parser.add_argument("-v", "--verbosity", action="count", default=0, help="increase output verbosity")

group = parser.add_mutually_exclusive_group()
group.add_argument("--save", help="save resulting model", type=str)
group.add_argument("--load", help="load saved model", type=str)

args = parser.parse_args()

with open(args.data, 'r') as input_file:
    words = input_file.read().split()
    
    instance = model.model(args.dimensions)
    train(instance)
