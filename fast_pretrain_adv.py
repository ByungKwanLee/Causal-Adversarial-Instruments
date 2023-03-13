# Import built-in module
import argparse
import warnings
warnings.filterwarnings(action='ignore')

# Import torch
import torch.optim as optim
import torch.distributed as dist

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
parser.add_argument('--NAME', default='ADV', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--network', default='vgg', type=str)
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--gpu', default='0,1,2,3', type=str)
parser.add_argument('--port', default="12355", type=str)
parser.add_argument('--load', default='false', type=str2bool)

# learning parameter
parser.add_argument('--learning_rate', default=0.1, type=float)
parser.add_argument('--weight_decay', default=0.0002, type=float)
parser.add_argument('--batch_size', default=128, type=float)
parser.add_argument('--test_batch_size', default=256, type=float)
parser.add_argument('--epoch', default=30, type=int)

# attack parameter only for CIFAR-10 and SVHN
parser.add_argument('--attack', default='pgd', type=str)
parser.add_argument('--eps', default=0.03, type=float)
parser.add_argument('--steps', default=10, type=int)
args = parser.parse_args()

# the number of gpus for multi-process
gpu_list = list(map(int, args.gpu.split(',')))
ngpus_per_node = len(gpu_list)

# cuda visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = args.port

# global best_acc
best_acc = 0

# Mix Training
scaler = GradScaler()

def train(net, trainloader, optimizer, lr_scheduler, scaler, attack):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    desc = ('[Train/LR=%.3f] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (lr_scheduler.get_lr()[0], 0, 0, correct, total))

    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs = attack(inputs, targets)

        # Accelerating forward propagation
        optimizer.zero_grad()
        with autocast():
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)

        # Accelerating backward propagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # scheduling for Cyclic LR
        lr_scheduler.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[Train/LR=%.3f] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (lr_scheduler.get_lr()[0], train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)



def test(net, testloader, attack, rank):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    desc = ('[Test/Clean] Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(0+1), 0, correct, total))

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=False)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()

        # Accerlating forward propagation
        with autocast():
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[Test/Clean] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    # Save clean acc.
    clean_acc = 100. * correct / total

    test_loss = 0
    correct = 0
    total = 0

    desc = ('[Test/PGD] Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss / (0 + 1), 0, correct, total))

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=False)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs = attack(inputs, targets)
        inputs, targets = inputs.cuda(), targets.cuda()

        # Accerlating forward propagation
        with autocast():
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[Test/PGD] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    # Save adv acc.
    adv_acc = 100. * correct / total

    # compute acc
    acc = (clean_acc + adv_acc)/2

    rprint('Current Accuracy is {:.2f}/{:.2f}!!'.format(clean_acc, adv_acc), rank)

    if acc > best_acc:
        state = {
            'net': net.state_dict(),
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if not os.path.isdir('checkpoint/pretrain'):
            os.mkdir('checkpoint/pretrain')

        best_acc = acc
        if rank == 0:
            torch.save(state, './checkpoint/pretrain/%s/%s_adv_%s%s_best.t7' % (args.dataset, args.dataset,
                                                                                args.network,
                                                                                args.depth))
            print('Saving~ ./checkpoint/pretrain/%s/%s_adv_%s%s_best.t7' % (args.dataset, args.dataset,
                                                                            args.network,
                                                                            args.depth))


def main_worker(rank, ngpus_per_node=ngpus_per_node):

    # print configuration
    print_configuration(args, rank)

    # setting gpu id of this process
    torch.cuda.set_device(rank)

    # DDP environment settings
    print("Use GPU: {} for training".format(gpu_list[rank]))
    dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=rank)

    # init model and Distributed Data Parallel
    net = get_network(network=args.network,
                      depth=args.depth,
                      dataset=args.dataset)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = net.to(memory_format=torch.channels_last).cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank], output_device=[rank])

    # fast init dataloader
    trainloader, testloader, decoder = get_fast_dataloader(dataset=args.dataset,
                                                           train_batch_size=args.batch_size,
                                                           test_batch_size=args.test_batch_size)

    # Load Plain Network
    if args.load:
        checkpoint_name = 'checkpoint/pretrain/%s/%s_%s%s_best.t7' % (args.dataset, args.dataset, args.network, args.depth)
        checkpoint = torch.load(checkpoint_name, map_location=torch.device(torch.cuda.current_device()))
        net.load_state_dict(checkpoint['net'])
        rprint(f'==> {checkpoint_name}', rank)
        rprint('==> Successfully Loaded Standard checkpoint..', rank)

    # Attack loader
    if args.dataset == 'imagenet':
        rprint('Fast FGSM training', rank)
        attack = attack_loader(net=net, attack='fgsm_train', eps=2/255, steps=args.steps)
    elif args.dataset == 'tiny':
        rprint('PGD and FGSM MIX training', rank)
        pgd_attack = attack_loader(net=net, attack='pgd', eps=4/255, steps=args.steps)
        fgsm_attack = attack_loader(net=net, attack='fgsm_train', eps=4/255, steps=args.steps)
        attack = MixAttack(net=net, slowattack=pgd_attack, fastattack=fgsm_attack, train_iters=len(trainloader))
    else:
        rprint('PGD training', rank)
        attack = attack_loader(net=net, attack=args.attack, eps=args.eps, steps=args.steps)

    # init optimizer and lr scheduler
    args.epoch = args.epoch if args.dataset !='tiny' else 4
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0, max_lr=args.learning_rate,
                                                     step_size_up=5 * len(trainloader) if args.dataset != 'imagenet' and args.dataset != 'tiny' else 2 * len(trainloader),
                                                     step_size_down=(args.epoch - 5) * len(trainloader) if args.dataset != 'imagenet' and args.dataset != 'tiny' else (args.epoch - 2) * len(trainloader))

    # training and testing
    for epoch in range(args.epoch):
        rprint('\nEpoch: %d' % (epoch+1), rank)
        if args.dataset == "imagenet":
            res = get_resolution(epoch=epoch, min_res=160, max_res=192, end_ramp=25, start_ramp=18)
            decoder.output_size = (res, res)
        train(net, trainloader, optimizer, lr_scheduler, scaler, attack)
        test(net, testloader, attack, rank)


def run():
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, join=True)

if __name__ == '__main__':
    run()