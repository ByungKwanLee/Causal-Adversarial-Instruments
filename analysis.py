from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import random

from tqdm import tqdm
from utils.fast_network_utils import get_network
from utils.fast_data_utils import get_fast_dataloader
from utils.utils import *

# attack loader
from attack.fastattack import attack_loader

# warning ignore
import warnings
warnings.filterwarnings("ignore")
from utils.utils import str2bool


# fetch args
parser = argparse.ArgumentParser()

# model parameter
parser.add_argument('--dataset', default='imagenet', type=str)
parser.add_argument('--network', default='resnet', type=str)
parser.add_argument('--depth', default=18, type=int)
parser.add_argument('--base', default='adv', type=str)
parser.add_argument('--batch_size', default=1, type=float)
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

# init dataloader
_, testloader = get_fast_dataloader(dataset=args.dataset, train_batch_size=1, test_batch_size=args.batch_size, gpu=args.gpu, dist=False)

# init model
net = get_network(network=args.network, depth=args.depth, dataset=args.dataset, gpu=int(args.gpu))
net = net.cuda()

# Load Plain Network
print('==> Loading Plain checkpoint..')
assert os.path.isdir('checkpoint/pretrain'), 'Error: no checkpoint directory found!'

# Loading checkpoint

if args.base == 'plain':
    checkpoint_name = 'checkpoint/pretrain/%s/%s_%s%s_best.t7' % (args.dataset, args.dataset, args.network, args.depth)
else:
    checkpoint_name = 'checkpoint/pretrain/%s/%s_%s_%s%s_best.t7' % (
    args.dataset, args.dataset, args.base, args.network, args.depth)

print("This analysis : {}".format(checkpoint_name))
checkpoint = torch.load(checkpoint_name, map_location=lambda storage, loc: storage.cuda())
net.load_state_dict(checkpoint['net'])

# init criterion
criterion = nn.CrossEntropyLoss()

def test():
    net.eval()
    test_loss = 0

    attack_score = []
    attack_module = {}
    # for attack_name in ['Plain', 'fgsm', 'pgd', 'cw_Linf', 'apgd', 'auto']:
    for attack_name in ['pgd']:
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

def visualizaition():
    v_seed = 777
    torch.manual_seed(v_seed)
    torch.cuda.manual_seed(v_seed)
    np.random.seed(v_seed)
    torch.random.manual_seed(v_seed)
    random.seed(v_seed)

    net.eval()

    if args.vis_atk:
        save_dir = './results/%s_vis_'%(str(args.attack)) + str(args.dataset) + '_' + str(args.network) + '_' + str(args.eps)
    else:
        save_dir = './results/clean_vis_' + str(args.dataset) + '_' + str(args.network) + '_' + str(args.eps)

    check_dir(save_dir)
    attack = attack_loader(net=net, attack='pgd', eps=args.eps, steps=args.steps,  dataset=args.dataset)

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()

        if args.vis_atk:
            inputs = attack(inputs, targets) if args.eps != 0 else inputs

        int_output = net(inputs, pop=True)

        print("ok")



if __name__ == '__main__':
    visualizaition()






