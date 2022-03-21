# Import built-in module
import os
import argparse
import warnings
warnings.filterwarnings(action='ignore')
import tracemalloc

# Import torch
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from tqdm import tqdm
from tensorboardX import SummaryWriter

# Import Custom Utils
from utils.fast_network_utils import get_network
from utils.fast_data_utils import get_fast_dataloader
from utils.utils import *

# attack loader
# from attack.attack import attack_loader
from attack.fastattack import attack_loader

# Accelerating forward and backward
from torch.cuda.amp import GradScaler, autocast

torch.backends.cudnn.benchmark = True
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)

# fetch args
parser = argparse.ArgumentParser()

# model parameter
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--network', default='vgg', type=str)

parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--gpu', default='0,1,2,3', type=str)
parser.add_argument('--pretrained', default=False, type=str2bool) # True for loading ImageNet pre-trained model

# learning parameter
parser.add_argument('--learning_rate', default=0.0001, type=float)
parser.add_argument('--weight_decay', default=0.0002, type=float)
parser.add_argument('--batch_size', default=128, type=float)
parser.add_argument('--test_batch_size', default=128, type=float)
parser.add_argument('--epoch', default=60, type=int)

# attack parameter
parser.add_argument('--attack', default='pgd', type=str)
parser.add_argument('--eps', default=0.03, type=float)
parser.add_argument('--steps', default=10, type=int)

parser.add_argument('--log_dir', type=str, default='logs', help='directory of training logs')
args = parser.parse_args()

# multi-process
ngpus_per_node = len(args.gpu.split(','))

# cuda visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# global best_acc
best_acc = 0

# LR Scheduler
# lr_schedule = {0: args.learning_rate,
#                int(args.epoch * 0.5): args.learning_rate * 0.1,
#                int(args.epoch * 0.75): args.learning_rate * 0.01}
# lr_scheduler = PresetLRScheduler(lr_schedule)

# init criterion
criterion = nn.CrossEntropyLoss()

# Mix Training
scaler = GradScaler()
counter = 0
log_dir = args.log_dir + '/'
check_dir(log_dir)
writer = SummaryWriter(log_dir=log_dir)

def causal_train(epoch, net, c_net, z_net, m_net, trainloader, c_optimizer, inst_optimizer, scaler, attack, gpu):
    global counter
    print('\nEpoch: %d' % epoch)

    net.eval()
    c_net.train()
    z_net.train()
    m_net.train()
    train_closs = 0
    train_zloss = 0
    train_celoss = 0
    correct = 0
    total = 0

    resize = get_resolution(epoch=epoch, min_res=160, max_res=192, end_ramp=48, start_ramp=41)

    c_scheduler = torch.optim.lr_scheduler.MultiStepLR(c_optimizer, milestones=[20, 40, 60], gamma=0.5)
    z_scheduler = torch.optim.lr_scheduler.MultiStepLR(inst_optimizer, milestones=[20, 40, 60], gamma=0.5)

    desc = ('[Train/C_LR=%s/Z_LR=%s] CELoss: %.3f | CLoss: %.3f | ZLoss: %.3f | Acc: %.3f%% (%d/%d)' %
            (c_scheduler.get_last_lr()[0], z_scheduler.get_last_lr()[0], 0, 0, 0, 0, correct, total))

    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.to(gpu), targets.to(gpu)

        if args.dataset == 'imagenet':
            inputs = resize(inputs)

        # epsilon = torch.empty_like(inputs).uniform_(-args.eps, args.eps).cuda()
        adv_inputs = attack(inputs, targets)
        #p_image = torch.clamp(inputs + epsilon, min=0, max=1).detach()

        c_optimizer.zero_grad(), inst_optimizer.zero_grad()

        # Accerlating forward propagation
        #snapshot1 = tracemalloc.take_snapshot()
        with autocast():
            pseudo_output = net(adv_inputs)
            pseudo_label, pseudo_predicted = get_pseudo(pseudo_output)

            inst_v = m_net(adv_inputs - inputs)
            cln_feature = net(inputs, pop=True)

            treat_feature = cln_feature + inst_v

            causal_feature = c_net(treat_feature)
            causal_output = net(causal_feature, int=True)

            inst_feature = z_net(inst_v)
            inst_output = net(inst_feature, int=True)

            causal_loss = ((pseudo_label - causal_output) * inst_output).mean()

        # Accerlating backward propagation

        scaler.scale(causal_loss).backward(retain_graph=True)
        scaler.step(c_optimizer)
        scaler.update()

        c_optimizer.zero_grad(), inst_optimizer.zero_grad()

        with autocast():
            causal_feature = c_net(treat_feature)
            causal_output = net(causal_feature, int=True)

            inst_loss = -1. * (((pseudo_label - causal_output) * inst_output).mean())
            ce_loss = criterion(causal_output, pseudo_predicted) # For XE loss checking
            
        # Accerlating backward propagation
        scaler.scale(inst_loss).backward()
        scaler.step(inst_optimizer)
        scaler.update()

        # snapshot2 = tracemalloc.take_snapshot()
        # top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        # print(top_stats[0])

        writer.add_scalar('Train/causal_loss', causal_loss, counter)
        writer.add_scalar('Train/inst_loss', inst_loss, counter)
        writer.add_scalar('Train/ce_loss', ce_loss, counter)
        writer.add_scalar('Train/lr', c_scheduler.get_last_lr()[0], counter)

        train_closs += causal_loss.item()
        train_zloss += inst_loss.item()
        train_celoss += ce_loss.item()
        counter += 1

        _, predicted = causal_output.max(1)
        total += pseudo_predicted.size(0)
        correct += predicted.eq(pseudo_predicted).sum().item()

        desc = ('[Train/C_LR=%s/Z_LR=%s] CELoss: %.3f | CLoss: %.3f | ZLoss: %.3f | Acc: %.3f%% (%d/%d)' %
                (c_scheduler.get_last_lr()[0], z_scheduler.get_last_lr()[0], train_celoss / (batch_idx + 1),
                 train_closs / (batch_idx + 1), train_zloss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

def causal_test(epoch, net, c_net, z_net, m_net, testloader, criterion, attack, gpu):
    global best_acc
    net.eval()
    c_net.eval()
    z_net.eval()
    m_net.eval()

    test_loss = 0
    correct = 0
    total = 0

    desc = ('[Test/PGD] Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss / (0 + 1), 0, correct, total))

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        adv_inputs = attack(inputs, targets)
        inputs, targets = inputs.to(gpu), targets.to(gpu)

        # Accerlating forward propagation
        with autocast():
            pseudo_output = net(adv_inputs)
            pseudo_label, pseudo_predicted = get_pseudo(pseudo_output)

            inst_v = m_net(adv_inputs - inputs)
            cln_feature = net(inputs, pop=True)

            treat_feature = cln_feature + inst_v

            causal_feature = c_net(treat_feature)
            causal_output = net(causal_feature, int=True)

            loss = criterion(causal_output, pseudo_predicted)

        test_loss += loss.item()
        _, predicted = causal_output.max(1)
        total += pseudo_predicted.size(0)
        correct += predicted.eq(pseudo_predicted).sum().item()

        desc = ('[Test/PGD] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)


    # Save adv acc.
    pseudo_acc = 100. * correct / total

    if pseudo_acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': pseudo_acc,
            'epoch': epoch,
            'loss': loss,
            'args': args
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if not os.path.isdir('checkpoint/pretrain'):
            os.mkdir('checkpoint/pretrain')

        if gpu == int(args.gpu.split(',')[0]):
            torch.save(state, './checkpoint/pretrain/%s/%s_adv_%s%s_best.t7' % (args.dataset, args.dataset,
                                                                         args.network,
                                                                         args.depth))
            print('./checkpoint/pretrain/%s/%s_adv_%s%s_best.t7' % (args.dataset, args.dataset,
                                                                         args.network,
                                                                         args.depth))
            best_acc = pseudo_acc

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
    net = get_network(network=args.network, depth=args.depth, dataset=args.dataset, gpu=gpu, pretrained=args.pretrained)
    net = net.to(memory_format=torch.channels_last).to(gpu)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpu])
    for params in net.parameters():
        params.requires_grad = False

    c_net = get_network(network='causal', depth=None, dataset=args.dataset, gpu=gpu, pretrained=None)
    c_net = c_net.to(memory_format=torch.channels_last).to(gpu)
    c_net = torch.nn.parallel.DistributedDataParallel(c_net, device_ids=[gpu])

    z_net = get_network(network='instrument', depth=args.depth, dataset=args.dataset,
                       gpu=gpu, pretrained=None, exo=True, exo_net=args.network)
    z_net = z_net.to(memory_format=torch.channels_last).to(gpu)
    z_net = torch.nn.parallel.DistributedDataParallel(z_net, device_ids=[gpu])

    m_net = get_network(network='instrument', depth=args.depth, dataset=args.dataset,
                        gpu=gpu, pretrained=None, exo=False, exo_net=args.network)
    m_net = m_net.to(memory_format=torch.channels_last).to(gpu)
    m_net = torch.nn.parallel.DistributedDataParallel(m_net, device_ids=[gpu])

    # fast init dataloader
    trainloader, testloader = get_fast_dataloader(dataset=args.dataset,
                                                  train_batch_size=args.batch_size,
                                                  test_batch_size=args.test_batch_size, gpu=gpu)

    #Load backbone network parameters
    print('==> Loading Backbone checkpoint..')
    assert os.path.isdir('checkpoint/pretrain'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('checkpoint/pretrain/%s/%s_%s%s_best.t7' % (args.dataset, args.dataset, args.network, args.depth))
    net.load_state_dict(checkpoint['net'])

    # Attack loader
    if args.dataset == 'imagenet':
        print('Fast FGSM training')
        attack = attack_loader(net=net, attack='fgsm_train', eps=args.eps, steps=args.steps, dataset=args.dataset)
    else:
        print('PGD training')
        attack = attack_loader(net=net, attack=args.attack, eps=args.eps, steps=args.steps, dataset=args.dataset)

    # init optimizer and lr scheduler
    # c_optimizer = optim.SGD(c_net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    # inst_optimizer = optim.SGD([{'params': z_net.parameters()}, {'params': m_net.parameters()}], lr=args.learning_rate,
    #                         momentum=0.9, weight_decay=args.weight_decay)
    c_optimizer = optim.AdamW(c_net.parameters(), lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=1e-4)
    inst_optimizer = optim.AdamW([{'params': z_net.parameters()}, {'params': m_net.parameters()}], lr=args.learning_rate,
                               betas=(0.5, 0.999), weight_decay=1e-4)
    #tracemalloc.start()
    for epoch in range(args.epoch):
        causal_train(epoch, net, c_net, z_net, m_net, trainloader, c_optimizer, inst_optimizer, scaler, attack, gpu)
        causal_test(epoch, net, c_net, z_net, m_net, testloader, criterion, attack, gpu)

    dist.destroy_process_group()

def run():
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, join=True)

if __name__ == '__main__':
    run()