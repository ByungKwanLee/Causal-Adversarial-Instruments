# Future import build
from __future__ import print_function

# Import built-in module
import os
import argparse

# fetch args
parser = argparse.ArgumentParser()

# parameter
parser.add_argument('--dataset', default='imagenet', type=str)
parser.add_argument('--gpu', default='0', type=str)
args = parser.parse_args()

# GPU configurations
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

# init fast dataloader
from utils.fast_data_utils import save_data_for_beton
save_data_for_beton(dataset=args.dataset)