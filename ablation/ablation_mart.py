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
parser.add_argument('--NAME', default='Ablation-MART', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--network', default='vgg', type=str)
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--gpu', default='0,1,2,3', type=str)
parser.add_argument('--port', default="12357", type=str)

# learning parameter
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--weight_decay', default=0.0002, type=float)
parser.add_argument('--batch_size', default=128, type=float)
parser.add_argument('--test_batch_size', default=256, type=float)
parser.add_argument('--epoch', default=4, type=int)

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


def train(net, c_net, trainloader, optimizer, lr_scheduler, scaler, attack):
    net.train()
    c_net.eval()
    train_loss = 0
    correct = 0
    total = 0

    desc = ('[Train/LR=%.3f] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (lr_scheduler.get_lr()[0], 0, 0, correct, total))

    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        adv_inputs = attack(inputs, targets)

        # Accerlating forward propagation
        optimizer.zero_grad()
        with autocast():

            # clean feature
            clean_feature = net(inputs, pop=True)
            clean_outputs = net(clean_feature.clone(), int=True)

            # adv feature
            adv_feature = net(adv_inputs, pop=True)
            adv_outputs = net(adv_feature.clone(), int=True)

            # causal feature and output
            del_causal = c_net(adv_feature - clean_feature)
            causal_feature = clean_feature + del_causal
            causal_outputs = net(causal_feature.clone(), int=True)

            # again inv causal feature for MART
            loss = mart_loss(clean_outputs, adv_outputs, targets) + causal_loss(adv_outputs, causal_outputs.detach())

        # Accerlating backward propagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # scheduling for Cyclic LR
        lr_scheduler.step()

        train_loss += loss.item()
        _, predicted = adv_outputs.max(1)
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
            torch.save(state, './checkpoint/ablation/%s/%s_ablationmart_%s%s_best.t7' % (args.dataset, args.dataset,
                                                                                args.network,
                                                                                args.depth))
            print('Saving~ ./checkpoint/ablation/%s/%s_ablationmart_%s%s_best.t7' % (args.dataset, args.dataset,
                                                                            args.network,
                                                                            args.depth))

def mart_loss(logits,
            logits_adv,
            targets):
    kl = torch.nn.KLDivLoss(reduction='none')
    adv_probs = F.softmax(logits_adv, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == targets, tmp1[:, -2], tmp1[:, -1])
    loss_adv = F.cross_entropy(logits_adv, targets) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
    nat_probs = F.softmax(logits, dim=1)
    true_probs = torch.gather(nat_probs, 1, (targets.unsqueeze(1)).long()).squeeze()
    loss_robust = (1.0 / logits.shape[0]) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss = loss_adv + float(1) * loss_robust
    return loss

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

    # init causal net and Distributed Data Parallel
    c_net = get_network(network='causal', depth=None, dataset=args.dataset, ch=True if args.network=='wide' else False)
    c_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(c_net)
    c_net = c_net.to(memory_format=torch.channels_last).cuda()
    c_net = torch.nn.parallel.DistributedDataParallel(c_net, device_ids=[rank], output_device=[rank])
    do_freeze(c_net)

    # fast init dataloader
    trainloader, testloader, decoder = get_fast_dataloader(dataset=args.dataset,
                                                  train_batch_size=args.batch_size,
                                                  test_batch_size=args.test_batch_size)

    # Load ADV Network
    checkpoint_name = 'checkpoint/pretrain/%s/%s_adv_%s%s_best.t7' % (args.dataset, args.dataset, args.network, args.depth)
    checkpoint = torch.load(checkpoint_name, map_location=torch.device(torch.cuda.current_device()))
    net.load_state_dict(checkpoint['net'])
    rprint(f'==> {checkpoint_name}', rank)
    rprint('==> Successfully Loaded Standard checkpoint..', rank)

    # Load Causal Network
    checkpoint_name = 'checkpoint/causal/%s/%s_causal_%s%s_best.t7' % (args.dataset, args.dataset, args.network, args.depth)
    checkpoint = torch.load(checkpoint_name, map_location=torch.device(torch.cuda.current_device()))
    c_net.load_state_dict(checkpoint['c_net'])
    rprint(f'==> {checkpoint_name}', rank)
    rprint('==> Successfully Loaded Causal checkpoint..', rank)

    # Attack loader
    if args.dataset == 'tiny':
        rprint('FGSM training', rank)
        attack = attack_loader(net=net, attack='fgsm_train', eps=4/255, steps=args.steps)
    else:
        rprint('PGD training', rank)
        attack = attack_loader(net=net, attack=args.attack, eps=args.eps, steps=args.steps)

    # init optimizer and lr scheduler
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0, max_lr=args.learning_rate,
    step_size_up=args.epoch * len(trainloader)/2, step_size_down=args.epoch * len(trainloader)/2)

    # training and testing
    for epoch in range(args.epoch):
        rprint('\nEpoch: %d' % (epoch+1), rank)
        train(net, c_net, trainloader, optimizer, lr_scheduler, scaler, attack)
        test(net, testloader, attack, rank)


def run():
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, join=True)

if __name__ == '__main__':
    run()