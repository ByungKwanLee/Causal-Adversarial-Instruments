# Future
from __future__ import print_function

# warning ignore
import warnings
warnings.filterwarnings("ignore")

import argparse
import torch.nn as nn
import torch.distributed as dist

from utils.fast_network_utils import get_network
from utils.fast_data_utils import get_fast_dataloader
from utils.utils import *

# fetch args
parser = argparse.ArgumentParser()

# model parameter
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--network', default='vgg', type=str)
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--base', default='plain', type=str)
parser.add_argument('--batch_size', default=128, type=float)
parser.add_argument('--gpu', default='0', type=str)

# attack parameter
parser.add_argument('--attack', default='pgd', type=str)
parser.add_argument('--eps', default=0.03, type=float)
parser.add_argument('--steps', default=30, type=int)
args = parser.parse_args()

# Printing configurations
print_configuration(args)

# GPU configurations
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

# the number of gpus for multi-process
gpu_list = list(map(int, args.gpu.split(',')))
ngpus_per_node = len(gpu_list)

def main_worker(rank, ngpus_per_node=ngpus_per_node):

    # print configuration
    print_configuration(args, rank)

    # setting gpu id of this process
    torch.cuda.set_device(rank)

    # DDP environment settings
    print("Use GPU: {} for training".format(gpu_list[rank]))
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=rank)


    # init model and Distributed Data Parallel
    net = get_network(network=args.network,
                      depth=args.depth,
                      dataset=args.dataset)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = net.to(memory_format=torch.channels_last).cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank], output_device=[rank])
    net.eval()


    # init dataloader
    _, testloader = get_fast_dataloader(dataset=args.dataset,
                                        train_batch_size=1,
                                        test_batch_size=args.batch_size)

    # Loading checkpoint
    if args.base == 'plain':
        checkpoint_name = 'checkpoint/pretrain/%s/%s_%s%s_best.t7' % (args.dataset, args.dataset, args.network, args.depth)
    else:
        checkpoint_name = 'checkpoint/pretrain/%s/%s_%s_%s%s_best.t7' % (
        args.dataset, args.dataset, args.base, args.network, args.depth)

    rprint("This test : {}".format(checkpoint_name), rank)
    checkpoint = torch.load(checkpoint_name)
    net.load_state_dict(checkpoint['net'])

    # init criterion
    criterion = nn.CrossEntropyLoss()

    # test
    test_robustness(net, testloader, criterion, attack_list=['Plain'], rank=rank)

    # destroy process
    dist.destroy_process_group()

def run():
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, join=True)

if __name__ == '__main__':
    run()






