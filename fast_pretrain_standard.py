# Future
from __future__ import print_function

# Built-in module
import os
import argparse

# torch pkg
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

# Cudnn settings
torch.backends.cudnn.benchmark = True
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)

# tqdm
from tqdm import tqdm

# Import Custom Utils
from utils.fast_network_utils import get_network
from utils.fast_data_utils import get_fast_dataloader
from utils.utils import *

# Accelerating forward and backward
from torch.cuda.amp import GradScaler, autocast


# fetch args
parser = argparse.ArgumentParser()


# model parameter
parser.add_argument('--dataset', default='imagenet', type=str)
parser.add_argument('--network', default='resnet', type=str)
parser.add_argument('--depth', default=50, type=int)
parser.add_argument('--gpu', default='0,1,2,3,4', type=str)

# learning parameter
parser.add_argument('--learning_rate', default=0.1, type=float)
parser.add_argument('--weight_decay', default=0.0002, type=float)
parser.add_argument('--batch_size', default=512, type=float)
parser.add_argument('--test_batch_size', default=128, type=float)
parser.add_argument('--epoch', default=100, type=int)
args = parser.parse_args()

# the number of gpus for multi process
ngpus_per_node = len(args.gpu.split(','))

# cuda visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# global best_acc
best_acc = 0


def train(epoch, net, trainloader, optimizer, criterion, lr_scheduler, scaler, gpu):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    resize = get_resolution(epoch=epoch, min_res=160, max_res=192, end_ramp=65, start_ramp=76)
    lr_scheduler(optimizer, epoch)
    desc = ('[Train/LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (lr_scheduler.get_lr(optimizer), 0, 0, correct, total))


    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.to(gpu), targets.to(gpu)
        inputs = resize(inputs)
        optimizer.zero_grad()

        # Accerlating forward propagation
        with autocast():
            outputs = net(inputs)
            loss = criterion(outputs, targets)

        # Accerlating backward propagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[Train/LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (lr_scheduler.get_lr(optimizer), train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)



def test(epoch, net, testloader, optimizer, criterion, lr_scheduler, gpu):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    desc = ('[Test/LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (lr_scheduler.get_lr(optimizer), test_loss/(0+1), 0, correct, total))

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.to(gpu), targets.to(gpu)

        # Accerlating forward propagation
        with autocast():
            outputs = net(inputs)
            loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[Test/LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (lr_scheduler.get_lr(optimizer), test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    # Save checkpoint.
    acc = 100.*correct/total


    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'loss': loss,
            'args': args
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if not os.path.isdir('checkpoint/pretrain'):
            os.mkdir('checkpoint/pretrain')

        if gpu == int(args.gpu.split(',')[0]):
            torch.save(state, './checkpoint/pretrain/%s/%s_%s%s_best.t7' % (args.dataset, args.dataset,
                                                                         args.network,
                                                                         args.depth))
            print('./checkpoint/pretrain/%s/%s_%s%s_best.t7' % (args.dataset, args.dataset,
                                                                         args.network,
                                                                         args.depth))
            best_acc = acc

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

    # init model and Distributed Data Parallel
    net = get_network(network=args.network,
                      depth=args.depth,
                      dataset=args.dataset,
                      gpu=gpu)
    net = net.to(memory_format=torch.channels_last).to(gpu)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpu])

    # fast dataloader
    trainloader, testloader = get_fast_dataloader(dataset=args.dataset,
                                                  train_batch_size=args.batch_size,
                                                  test_batch_size=args.test_batch_size, gpu=gpu)


    # init optimizer and lr scheduler
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    lr_schedule = {0: args.learning_rate,
                   int(args.epoch * 0.5): args.learning_rate * 0.1,
                   int(args.epoch * 0.75): args.learning_rate * 0.01}
    lr_scheduler = PresetLRScheduler(lr_schedule)
    scaler = GradScaler()

    # init criterion
    criterion = nn.CrossEntropyLoss()

    # training and testing
    for epoch in range(args.epoch):
        train(epoch, net, trainloader, optimizer, criterion, lr_scheduler, scaler, gpu)
        test(epoch, net, testloader, optimizer, criterion, lr_scheduler, gpu)

    dist.destroy_process_group()


def run():
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, join=True)

if __name__ == '__main__':
    run()