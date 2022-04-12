# Import built-in module
import argparse
import warnings

warnings.filterwarnings(action='ignore')

import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
# Import Custom Utils
from utils.fast_network_utils import get_network
from utils.fast_data_utils import get_fast_dataloader
from utils.utils import *

# attack loader
from attack.fastattack import attack_loader

# Accelerating forward and backward
from torch.cuda.amp import GradScaler, autocast

torch.backends.cudnn.benchmark = True
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)

# fetch args
parser = argparse.ArgumentParser()

# model parameter
parser.add_argument('--dataset', default='imagenet', type=str)
parser.add_argument('--network', default='vgg', type=str)

parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--gpu', default='0,1,2,3', type=str)
parser.add_argument('--port', default='12355', type=str)

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

# the number of gpus for multi-process
gpu_list = list(map(int, args.gpu.split(',')))
ngpus_per_node = len(gpu_list)

# cuda visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = args.port

# global best_acc
best_acc = 0

# init criterion
criterion = nn.CrossEntropyLoss()
softmax = nn.Softmax(dim=1)

# Mix Training
scaler = GradScaler()
counter = 0
log_dir = args.log_dir + '/'
check_dir(log_dir)


def causal_train(epoch, net, c_net, z_net, trainloader, c_optimizer, inst_optimizer, c_scheduler, z_scheduler, scaler,
                 attack, rank, writer):
    global counter
    net.eval()
    c_net.train()
    z_net.train()

    adv_correct, inst_correct, treat_correct, causal_correct = 0, 0, 0, 0
    total = 0

    resize = get_resolution(epoch=epoch, min_res=160, max_res=192, end_ramp=27, start_ramp=23)

    desc = ('[Train/C_LR=%s/Z_LR=%s] Adv: %.1f%% | Inst: %.1f%% | Treat: %.1f%% | Causal: %.1f%%' %
            (c_scheduler.get_last_lr()[0], z_scheduler.get_last_lr()[0], 0, 0, 0, 0))

    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()

        if args.dataset == 'imagenet':
            inputs = resize(inputs)

        adv_inputs = attack(inputs, targets)

        c_optimizer.zero_grad(), inst_optimizer.zero_grad()

        # Accerlating forward propagation
        with autocast():
            adv_feature = net(adv_inputs, pop=True)
            cln_feature = net(inputs, pop=True)
            residual = adv_feature - cln_feature

            adv_output = net(adv_feature.clone().detach(), int=True)
            onehot_target = get_onehot(adv_output, targets)

            inst_feature = z_net(residual)
            inst_output = net(inst_feature.clone(), int=True)

            treat_feature = cln_feature + inst_feature
            treat_output = net(treat_feature.clone().detach(), int=True)

            causal_feature = c_net(treat_feature)
            causal_output = net(causal_feature.clone(), int=True)

            recon_loss = ((causal_feature - adv_feature.detach()) ** 2).mean()

            causal_loss = (onehot_target * F.log_softmax(causal_output) * F.log_softmax(inst_output)).sum(dim=1).mean() + recon_loss

        # Accerlating backward propagation
        scaler.scale(causal_loss).backward(retain_graph=True)
        scaler.step(c_optimizer)
        scaler.update()

        c_optimizer.zero_grad(), inst_optimizer.zero_grad()

        with autocast():
            causal_feature = c_net(treat_feature)
            causal_output = net(causal_feature.clone().detach(), int=True)

            reg = (inst_feature ** 2).mean()

            inst_loss = -(onehot_target * F.log_softmax(causal_output) * F.log_softmax(inst_output)).sum(dim=1).mean() + reg

            ce_loss = criterion(causal_output, targets)  # For XE loss checking
            ce_loss2 = criterion(inst_output, targets)  # For XE loss checking

        # Accerlating backward propagation
        scaler.scale(inst_loss).backward()
        scaler.step(inst_optimizer)
        scaler.update()
        if rank == 0:
            writer.add_scalar('Train/causal_loss', causal_loss, counter)
            writer.add_scalar('Train/inst_loss', inst_loss, counter)
            writer.add_scalar('Train/causlXE_loss', ce_loss, counter)
            writer.add_scalar('Train/instXE_loss', ce_loss2, counter)
            writer.add_scalar('Train/recon_loss', recon_loss, counter)
            writer.add_scalar('Train/lr', c_scheduler.get_last_lr()[0], counter)
            counter += 1

        _, adv_predicted = adv_output.max(1)
        _, inst_predicted = inst_output.max(1)
        _, treat_predicted = treat_output.max(1)
        _, causal_predicted = causal_output.max(1)

        total += targets.size(0)
        adv_correct += adv_predicted.eq(targets).sum().item()
        inst_correct += inst_predicted.eq(targets).sum().item()
        treat_correct += treat_predicted.eq(targets).sum().item()
        causal_correct += causal_predicted.eq(targets).sum().item()

        desc = ('[Train/C_LR=%s/Z_LR=%s] Adv: %.1f%% | Inst: %.1f%% | Treat: %.1f%% | Causal: %.1f%%' %
                (c_scheduler.get_last_lr()[0], z_scheduler.get_last_lr()[0], 100. * adv_correct / total,
                 100. * inst_correct / total, 100. * treat_correct / total, 100. * causal_correct / total))

        prog_bar.set_description(desc, refresh=True)


def causal_test(epoch, net, c_net, z_net, testloader, criterion, attack, rank):
    global best_acc
    net.eval()
    c_net.eval()
    z_net.eval()

    test_loss = 0
    correct = 0
    total = 0

    desc = ('[Test/PGD] Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss / (0 + 1), 0, correct, total))

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        adv_inputs = attack(inputs, targets)
        inputs, targets = inputs.cuda(), targets.cuda()

        # Accerlating forward propagation
        with autocast():
            adv_feature = net(adv_inputs, pop=True)
            cln_feature = net(inputs, pop=True)

            inst_feature = z_net(adv_feature - cln_feature)
            cln_feature = net(inputs, pop=True)

            treat_feature = cln_feature + inst_feature

            causal_feature = c_net(treat_feature)
            causal_output = net(causal_feature, int=True)

            loss = criterion(causal_output, targets)

        test_loss += loss.item()
        _, predicted = causal_output.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[Test/PGD] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    # Save adv acc.
    pseudo_acc = 100. * correct / total

    if pseudo_acc > best_acc:
        state = {
            'c_net': c_net.state_dict(),
            'z_net': z_net.state_dict(),
            'acc': pseudo_acc,
            'epoch': epoch,
            'loss': loss,
            'args': args
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if not os.path.isdir('checkpoint/pretrain'):
            os.mkdir('checkpoint/pretrain')
        best_acc = pseudo_acc

        if rank == 0:
            torch.save(state, './checkpoint/pretrain/%s/%s_causal_causal_reg_%s%s_best.t7' % (
            args.dataset, args.dataset, args.network, args.depth))
            print('Saving~ ./checkpoint/pretrain/%s/%s_causal_recon_%s%s_best.t7' % (
            args.dataset, args.dataset, args.network, args.depth))


def main_worker(rank, ngpus_per_node=ngpus_per_node):
    # print configuration
    print_configuration(args, rank)

    # setting gpu id of this process
    torch.cuda.set_device(rank)

    # DDP environment settings
    print("Use GPU: {} for training".format(gpu_list[rank]))
    dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=rank)

    # init model and Distributed Data Parallel
    net = get_network(network=args.network, depth=args.depth, dataset=args.dataset)
    net = net.to(memory_format=torch.channels_last).cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank], output_device=[rank])
    do_freeze(net)

    c_net = get_network(network='causal', depth=None, dataset=args.dataset)
    c_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(c_net)
    c_net = c_net.to(memory_format=torch.channels_last).cuda()
    c_net = torch.nn.parallel.DistributedDataParallel(c_net, device_ids=[rank], output_device=[rank])

    z_net = get_network(network='instrument', depth=args.depth, dataset=args.dataset)
    z_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(z_net)
    z_net = z_net.to(memory_format=torch.channels_last).cuda()
    z_net = torch.nn.parallel.DistributedDataParallel(z_net, device_ids=[rank], output_device=[rank])

    # fast init dataloader
    trainloader, testloader = get_fast_dataloader(dataset=args.dataset,
                                                  train_batch_size=args.batch_size,
                                                  test_batch_size=args.test_batch_size)

    # Load backbone network parameters
    rprint('==> Loading Backbone checkpoint..', rank)
    checkpoint = torch.load(
        'checkpoint/pretrain/%s/%s_adv_%s%s_best.t7' % (args.dataset, args.dataset, args.network, args.depth))
    net.load_state_dict(checkpoint['net'])

    # Attack loader
    if args.dataset == 'imagenet':
        rprint('Fast FGSM training', rank)
        attack = attack_loader(net=net, attack='fgsm_train', eps=args.eps, steps=args.steps)
    else:
        rprint('PGD training', rank)
        attack = attack_loader(net=net, attack=args.attack, eps=args.eps, steps=args.steps)

    # init optimizer and lr scheduler
    c_optimizer = optim.AdamW([{'params': c_net.parameters()}], lr=args.learning_rate,
                              betas=(0.5, 0.999), weight_decay=1e-4)
    inst_optimizer = optim.AdamW([{'params': z_net.parameters()}], lr=args.learning_rate,
                                 betas=(0.5, 0.999), weight_decay=1e-4)

    # tensorboard writer
    writer = SummaryWriter(log_dir=log_dir) if rank == 0 else None

    c_scheduler = torch.optim.lr_scheduler.MultiStepLR(c_optimizer, milestones=[20, 40, 60], gamma=0.5)
    z_scheduler = torch.optim.lr_scheduler.MultiStepLR(inst_optimizer, milestones=[20, 40, 60], gamma=0.5)

    for epoch in range(args.epoch):
        rprint('\nEpoch: %d' % epoch, rank)
        causal_train(epoch, net, c_net, z_net, trainloader, c_optimizer, inst_optimizer, c_scheduler, z_scheduler,
                     scaler, attack, rank, writer)
        causal_test(epoch, net, c_net, z_net, testloader, criterion, attack, rank)

    # destroy process
    dist.destroy_process_group()


def run():
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, join=True)


if __name__ == '__main__':
    run()