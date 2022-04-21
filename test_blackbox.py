# Future
from __future__ import print_function

# warning ignore
import warnings
warnings.filterwarnings("ignore")

import argparse
import torch.distributed as dist

from utils.fast_network_utils import get_network
from utils.fast_data_utils import get_fast_dataloader
from utils.utils import *

# fetch args
parser = argparse.ArgumentParser()

# model parameter
parser.add_argument('--dataset', default='cifar100', type=str)
parser.add_argument('--network', default='wide', type=str)
parser.add_argument('--depth', default=34, type=int)
parser.add_argument('--base', default='adv', type=str)
parser.add_argument('--batch_size', default=128, type=float)
parser.add_argument('--gpu', default='0', type=str) # necessarily one gpu id!!!!
parser.add_argument('--port', default='12355', type=str)

args = parser.parse_args()

# GPU configurations
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = args.port

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
    dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=rank)


    # init model and Distributed Data Parallel
    adv_net = get_network(network=args.network,
                      depth=args.depth,
                      dataset=args.dataset)
    adv_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(adv_net)
    adv_net = adv_net.to(memory_format=torch.channels_last).cuda()
    adv_net = torch.nn.parallel.DistributedDataParallel(adv_net, device_ids=[rank], output_device=[rank])
    adv_net.eval()


    # init model and Distributed Data Parallel
    plain_net = get_network(network=args.network,
                      depth=args.depth,
                      dataset=args.dataset)
    plain_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(plain_net)
    plain_net = plain_net.to(memory_format=torch.channels_last).cuda()
    plain_net = torch.nn.parallel.DistributedDataParallel(plain_net, device_ids=[rank], output_device=[rank])
    plain_net.eval()


    # init dataloader
    _, testloader, _ = get_fast_dataloader(dataset=args.dataset,
                                        train_batch_size=1,
                                        test_batch_size=args.batch_size)

    # Loading checkpoint
    plain_checkpoint_name = 'checkpoint/pretrain/%s/%s_%s%s_best.t7' % (args.dataset, args.dataset, args.network, args.depth)
    adv_checkpoint_name = 'checkpoint/pretrain/%s/%s_%s_%s%s_best.t7' % (args.dataset, args.dataset, args.base, args.network, args.depth)

    rprint("This test : {}".format(plain_checkpoint_name), rank)
    rprint("This test : {}".format(adv_checkpoint_name), rank)

    plain_checkpoint = torch.load(plain_checkpoint_name)
    adv_checkpoint = torch.load(adv_checkpoint_name)

    plain_net.load_state_dict(plain_checkpoint['net'])
    adv_net.load_state_dict(adv_checkpoint['net'])

    # test
    test_blackbox(plain_net, adv_net, testloader, attack_list=['fgsm', 'bim', 'pgd', 'mim', 'cw_linf', 'fab', 'ap', 'dlr', 'auto'], rank=rank)

    # destroy process
    dist.destroy_process_group()

def run():
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, join=True)

if __name__ == '__main__':
    run()






