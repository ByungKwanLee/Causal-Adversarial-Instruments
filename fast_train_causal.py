# Import built-in module

import argparse
import warnings
import math

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
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--network', default='vgg', type=str)

parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--gpu', default='0,1,2,3', type=str)
parser.add_argument('--port', default='12351', type=str)

# learning parameter
parser.add_argument('--learning_rate', default=0.0001, type=float)
parser.add_argument('--weight_decay', default=0.00001, type=float)
parser.add_argument('--batch_size', default=128, type=float)
parser.add_argument('--test_batch_size', default=256, type=float)
parser.add_argument('--epoch', default=10, type=int)
parser.add_argument('--lamb', default=1, type=int)

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
log_dir = args.log_dir + f'{args.lamb}/'
check_dir(log_dir)

def causal_train(epoch, net, c_net, z_net, trainloader, c_optimizer, inst_optimizer, scaler, attack, rank, writer):
    global counter
    net.eval()
    c_net.train()
    z_net.train()

    adv_correct, inst_correct, treat_correct, causal_correct = 0, 0, 0, 0
    total = 0

    desc = ('[Train] Adv: %.1f%% | Inst: %.1f%% | Treat: %.1f%% | Causal: %.1f%%' %(0, 0, 0, 0))

    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        adv_inputs = attack(inputs, targets)

        c_optimizer.zero_grad(), inst_optimizer.zero_grad()

        with autocast():
            adv_feature = net(adv_inputs, pop=True)
            cln_feature = net(inputs, pop=True)
            residual = adv_feature - cln_feature

            adv_output = net(adv_feature.clone(), int=True)
            onehot_target = get_onehot(adv_output, targets)

            inst_feature = z_net(residual.clone())
            inst_output = net(cln_feature.clone() + inst_feature.clone(), int=True)

            causal_feature = c_net(inst_feature)
            causal_output = net((cln_feature.clone() + causal_feature.clone()), int=True)
            treat_output = net(cln_feature.clone() + c_net(residual).clone(), int=True)

            reg_loss = args.lamb * ((inst_feature - residual) ** 2).mean()
            inst_loss = -(onehot_target * F.log_softmax(causal_output) * F.log_softmax(inst_output)).sum(dim=1).mean()
            max_total_loss = inst_loss + reg_loss

        # Accerlating backward propagation
        scaler.scale(max_total_loss).backward()
        scaler.step(inst_optimizer)

        c_optimizer.zero_grad(), inst_optimizer.zero_grad()

        # Accerlating forward propagation
        with autocast():
            inst_feature = z_net(residual)
            inst_output = net(cln_feature.clone() + inst_feature.clone(), int=True)

            causal_feature = c_net(inst_feature)
            causal_output = net(cln_feature.clone() + causal_feature.clone(), int=True)

            recon_loss = ((c_net(residual) - residual) ** 2).mean()
            causal_loss = (onehot_target * F.log_softmax(causal_output) * F.log_softmax(inst_output)).sum(dim=1).mean()
            min_total_loss = causal_loss

            ce_loss = criterion(causal_output, targets)  # For XE loss checking
            ce_loss2 = criterion(inst_output, targets)  # For XE loss checking
            ce_loss3 = criterion(treat_output, targets)  # For XE loss checking

        # Accerlating backward propagation
        scaler.scale(min_total_loss).backward(retain_graph=True)
        scaler.step(c_optimizer)
        scaler.update()

        if rank == 0:
            writer.add_scalar('Train_Loss/causal_loss', causal_loss, counter)
            writer.add_scalar('Train_Loss/inst_loss', inst_loss, counter)
            writer.add_scalar('Train_Loss/reg_loss', reg_loss, counter)
            writer.add_scalar('Train_Loss/recon_loss', recon_loss, counter)
            writer.add_scalar('XE_Loss/causlXE_loss', ce_loss, counter)
            writer.add_scalar('XE_Loss/instXE_loss', ce_loss2, counter)
            writer.add_scalar('XE_Loss/advXE_loss', ce_loss3, counter)
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

        desc = ('[Train] Adv: %.1f%% | Inst: %.1f%% | Treat: %.1f%% | Causal: %.1f%%' %(100. * adv_correct / total,
                                                                                        100. * inst_correct / total,
                                                                                        100. * treat_correct / total,
                                                                                        100. * causal_correct / total))
        prog_bar.set_description(desc, refresh=True)

def causal_test(epoch, net, c_net, z_net, testloader, criterion, attack, rank):
    global best_acc
    net.eval()
    c_net.eval()
    z_net.eval()

    test_loss = 0
    adv_correct, inst_correct, treat_correct, causal_correct = 0, 0, 0, 0
    total = 0

    desc = ('[Test/PGD] Loss: %.3f | Adv: %.1f%% | Inst: %.1f%% | Treat: %.1f%% | Causal: %.1f%% (%d/%d)'
            % (test_loss / (0 + 1), 0, adv_correct, inst_correct, treat_correct, causal_correct, total))

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        adv_inputs = attack(inputs, targets)
        inputs, targets = inputs.cuda(), targets.cuda()

        # Accerlating forward propagation
        with autocast():
            adv_feature = net(adv_inputs, pop=True)
            cln_feature = net(inputs, pop=True)
            residual = adv_feature - cln_feature

            inst_feature = z_net(residual)
            treat_feature = c_net(residual)
            causal_feature = c_net(inst_feature)

            adv_output = net(adv_feature.clone(), int=True)
            inst_output = net(cln_feature.clone() + inst_feature.clone(), int=True)
            treat_output = net(cln_feature.clone() + treat_feature.clone(), int=True)
            causal_output = net(cln_feature.clone() + causal_feature.clone(), int=True)
            loss = criterion(treat_output, targets)

        test_loss += loss.item()
        _, adv_predicted = adv_output.max(1)
        _, inst_predicted = inst_output.max(1)
        _, treat_predicted = treat_output.max(1)
        _, causal_predicted = causal_output.max(1)

        total += targets.size(0)
        adv_correct += adv_predicted.eq(targets).sum().item()
        inst_correct += inst_predicted.eq(targets).sum().item()
        treat_correct += treat_predicted.eq(targets).sum().item()
        causal_correct += causal_predicted.eq(targets).sum().item()

        desc = ('[Test/PGD] Loss: %.3f | Adv: %.1f%% | Inst: %.1f%% | Treat: %.1f%% | Causal: %.1f%% (%d/%d)'
                % (test_loss / (batch_idx + 1), 100. * adv_correct / total, 100. * inst_correct / total,
                   100. * treat_correct / total, 100. * causal_correct / total, treat_correct, total))
        prog_bar.set_description(desc, refresh=True)

    # Save adv acc.
    state = {
        'c_net': c_net.state_dict(),
        'z_net': z_net.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'args': args
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir('checkpoint/pretrain'):
        os.mkdir('checkpoint/pretrain')

    if rank == 0:
        torch.save(state, './checkpoint/pretrain/%s/causal/%s_causal_%d_%s%s_E%d_best.t7' % (
        args.dataset, args.dataset, args.lamb, args.network, args.depth, epoch))

        print('Saving~ ./checkpoint/pretrain/%s/causal/%s_causal_%d_%s%s_best_E%d.t7' % (
        args.dataset, args.dataset, args.lamb, args.network, args.depth, epoch))

def main_worker(rank, ngpus_per_node=ngpus_per_node):
    # print configuration
    print_configuration(args, rank)

    # setting gpu id of this process
    torch.cuda.set_device(rank)

    # DDP environment settings
    print("Use GPU: {} for training".format(gpu_list[rank]))
    dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=rank)

    if args.network == 'wide':
        ch = True
    else:
        ch = False

    # init model and Distributed Data Parallel
    net = get_network(network=args.network, depth=args.depth, dataset=args.dataset)
    net = net.to(memory_format=torch.channels_last).cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank], output_device=[rank])
    do_freeze(net)

    # causal response network: hypothesis model
    c_net = get_network(network='causal', depth=None, dataset=args.dataset, ch=ch)
    c_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(c_net)
    c_net = c_net.to(memory_format=torch.channels_last).cuda()
    c_net = torch.nn.parallel.DistributedDataParallel(c_net, device_ids=[rank], output_device=[rank])

    # instrument Z network: test function
    z_net = get_network(network='instrument', depth=args.depth, dataset=args.dataset, ch=ch)
    z_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(z_net)
    z_net = z_net.to(memory_format=torch.channels_last).cuda()
    z_net = torch.nn.parallel.DistributedDataParallel(z_net, device_ids=[rank], output_device=[rank])

    # fast init dataloader
    trainloader, testloader, decoder = get_fast_dataloader(dataset=args.dataset,
                                                  train_batch_size=args.batch_size,
                                                  test_batch_size=args.test_batch_size)

    # Load backbone network parameters
    rprint('==> Loading Backbone checkpoint..', rank)
    checkpoint = torch.load(
        'checkpoint/pretrain/%s/%s_adv_%s%s_best.t7' % (args.dataset, args.dataset, args.network, args.depth), map_location=torch.device(torch.cuda.current_device()))
    net.load_state_dict(checkpoint['net'])

    # Attack loader
    if args.dataset == 'imagenet':
        rprint('Fast FGSM training', rank)
        attack = attack_loader(net=net, attack='fgsm_train', eps=2/255 if args.dataset == 'imagenet' else 4/255, steps=args.steps)
    elif args.network == 'wide' and args.dataset == 'tiny':
        rprint('PGD and FGSM MIX training', rank)
        pgd_attack = attack_loader(net=net, attack='pgd', eps=4/255, steps=args.steps)
        fgsm_attack = attack_loader(net=net, attack='fgsm_train', eps=4/255, steps=args.steps)
        attack = MixAttack(net=c_net, slowattack=pgd_attack, fastattack=fgsm_attack, train_iters=len(trainloader))
    elif args.dataset == 'tiny':
        rprint('PGD training', rank)
        attack = attack_loader(net=net, attack='pgd', eps=4/255, steps=args.steps)
    else:
        rprint('PGD training', rank)
        attack = attack_loader(net=net, attack=args.attack, eps=args.eps, steps=args.steps)

    # init optimizer and lr scheduler
    if args.dataset == 'imagenet' or args.dataset == 'tiny':
        c_optimizer = optim.AdamW([{'params': c_net.parameters()}], lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=1e-4)
        inst_optimizer = optim.AdamW([{'params': z_net.parameters()}], lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=1e-4)
    else:
        c_optimizer = optim.AdamW([{'params': c_net.parameters()}], lr=args.learning_rate,  weight_decay=args.weight_decay, amsgrad=True)
        inst_optimizer = optim.AdamW([{'params': z_net.parameters()}], lr=args.learning_rate,  weight_decay=args.weight_decay, amsgrad=True)

    # tensorboard writer
    writer = SummaryWriter(log_dir=log_dir) if rank == 0 else None

    for epoch in range(args.epoch):
        rprint('\nEpoch: %d' % epoch, rank)
        if args.dataset == "imagenet":
            res = get_resolution(epoch=epoch, min_res=160, max_res=192, end_ramp=25, start_ramp=18)
            decoder.output_size = (res, res)

        causal_train(epoch, net, c_net, z_net, trainloader, c_optimizer, inst_optimizer,scaler, attack, rank, writer)
        causal_test(epoch, net, c_net, z_net, testloader, criterion, attack, rank)


def run():
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, join=True)


if __name__ == '__main__':
    run()