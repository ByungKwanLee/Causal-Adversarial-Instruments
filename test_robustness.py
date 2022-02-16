# Future
from __future__ import print_function

# warning ignore
import warnings
warnings.filterwarnings("ignore")

import argparse
import torch
import torch.nn as nn
import torch.distributed as dist

from tqdm import tqdm
from utils.fast_network_utils import get_network
from utils.fast_data_utils import get_fast_dataloader
from utils.utils import *

# attack loader
from attack.fastattack import attack_loader

# fetch args
parser = argparse.ArgumentParser()

# model parameter
parser.add_argument('--dataset', default='imagenet', type=str)
parser.add_argument('--network', default='resnet', type=str)
parser.add_argument('--depth', default=50, type=int)
parser.add_argument('--base', default='plain', type=str)
parser.add_argument('--batch_size', default=128, type=float)
parser.add_argument('--gpu', default='0', type=str)

# attack parameter
parser.add_argument('--attack', default='pgd', type=str)
parser.add_argument('--eps', default=0.03, type=float)
parser.add_argument('--steps', default=30, type=int)
args = parser.parse_args()

# visualization parameter
parser.add_argument('--vis_attack', default='true', type=str2bool)

# Printing configurations
print_configuration(args)

# GPU configurations
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

# the number of gpus for multi process
ngpus_per_node = len(args.gpu.split(','))

# attack list
attack_list = ['Plain']


def test(net, testloader, criterion):
    net.eval()
    test_loss = 0

    attack_score = []
    attack_module = {}
    for attack_name in attack_list:
        args.attack = attack_name
        attack_module[attack_name] = attack_loader(net=net, attack=attack_name,
                                                   eps=args.eps, steps=args.steps,
                                                   dataset=args.dataset) \
            if attack_name != 'Plain' else None


    for key in attack_module:
        total = 0
        correct = 0
        prog_bar = tqdm(enumerate(testloader), total=len(testloader), leave=True)
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.cuda(), targets.cuda()
            if key != 'Plain':
                inputs = attack_module[key](inputs, targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('[Test/%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (key, test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

            # fast eval
            if (key == 'apgd') or (key == 'auto') or (key == 'cw_Linf') or (key == 'cw'):
                if batch_idx >= int(len(testloader) * 0.3):
                    break

        attack_score.append(100. * correct / total)

    print('\n----------------Summary----------------')
    print(args.steps, ' steps attack')
    for key, score in zip(attack_module, attack_score):
        print(str(key), ' : ', str(score) + '(%)')
    print('---------------------------------------\n')

def main_worker(gpu, ngpus_per_node=ngpus_per_node):


    if gpu == int(args.gpu.split(',')[0]):
        # Printing configurations
        print_configuration(args)
        print('==> Making model..')

    print("Use GPU: {} for training".format(gpu))
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=gpu)
    torch.cuda.set_device(gpu)


    # init model
    net = get_network(network=args.network,
                      depth=args.depth,
                      dataset=args.dataset,
                      gpu=gpu)
    net = net.to(memory_format=torch.channels_last).to(gpu)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpu])
    net.eval()


    # init dataloader
    _, testloader = get_fast_dataloader(dataset=args.dataset,
                                        train_batch_size=1,
                                        test_batch_size=args.batch_size,
                                        gpu=gpu)


    # Load Plain Network
    print('==> Loading Plain checkpoint..')
    assert os.path.isdir('checkpoint/pretrain'), 'Error: no checkpoint directory found!'

    # Loading checkpoint
    if args.base == 'plain':
        checkpoint_name = 'checkpoint/pretrain/%s/%s_%s%s_best.t7' % (args.dataset, args.dataset, args.network, args.depth)
    else:
        checkpoint_name = 'checkpoint/pretrain/%s/%s_%s_%s%s_best.t7' % (
        args.dataset, args.dataset, args.base, args.network, args.depth)

    print("This test : {}".format(checkpoint_name))
    checkpoint = torch.load(checkpoint_name)
    net.load_state_dict(checkpoint['net'])

    # init criterion
    criterion = nn.CrossEntropyLoss()

    # test
    test(net, testloader, criterion)

def run():
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, join=True)

if __name__ == '__main__':
    run()






