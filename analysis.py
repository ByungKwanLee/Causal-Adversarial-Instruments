from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn

from tqdm import tqdm
from utils.fast_network_utils import get_network
from utils.fast_data_utils import get_fast_dataloader
from utils.utils import *
from utils.visual_utils import *

# attack loader
from attack.fastattack import attack_loader

# warning ignore
import warnings
warnings.filterwarnings("ignore")
from utils.utils import str2bool

# fetch args
parser = argparse.ArgumentParser()

# model parameter
parser.add_argument('--dataset', default='cifar10', type=str) #imagenet cifar10 svhn tiny
parser.add_argument('--network', default='vgg', type=str)
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--base', default='adv', type=str)
parser.add_argument('--batch_size', default=1, type=float)
parser.add_argument('--gpu', default='0', type=str)

# attack parameter
parser.add_argument('--attack', default='pgd', type=str)
parser.add_argument('--eps', default=0.03, type=float)
parser.add_argument('--steps', default=30, type=int)

# visualization parameter
parser.add_argument('--f_type', default='combine', type=str, help='option: posneg / single / combine')

args = parser.parse_args()

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
    checkpoint_name = 'checkpoint/pretrain/%s/%s_%s_variation_%s%s_best.t7' % (
    args.dataset, args.dataset, args.base, args.network, args.depth)

print("This analysis : {}".format(checkpoint_name))
checkpoint = torch.load(checkpoint_name, map_location=lambda storage, loc: storage.cuda())['net']

from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
net.load_state_dict(new_state_dict)

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
                                                   eps=args.eps, steps=args.steps) if attack_name != 'Plain' else None


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
    net.eval()

    if args.base == 'plain':
        save_dir = './results/feature_vis/clean_vis_' + str(args.dataset) + '_' + str(args.network)
    else:
        save_dir = './results/feature_vis/%s_vis_' % (str(args.attack)) + str(args.dataset) + '_' + str(args.network) + '_' + str(args.eps)

    check_dir(save_dir)
    attack = attack_loader(net=net, attack='pgd', eps=args.eps, steps=args.steps)

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()

        if args.base != 'plain':
            inputs = attack(inputs, targets) if args.eps != 0 else inputs

        int_latent = net(inputs, pop=True)
        inv = SpInversion(int_latent.clone(), net, dataset=args.dataset).invert(inputs).squeeze()

        _, pred = net(int_latent, int=True).max(1)
        label = [targets.item(), pred.item()]

        save_img = feature_vis(inputs, inv, label, dataset=args.dataset)
        save_img.save(save_dir + '/inv_img%d.png' % (batch_idx))
        print("\n [*] Inversion Img%d is saved" % (batch_idx))

def net_visualize():
    net.eval()

    if args.base == 'plain':
        save_dir = './results/net_vis/%s/clean_vis_' % (str(args.f_type)) + str(args.dataset) + '_' + str(args.network)
    else:
        save_dir = './results/net_vis/%s/%s_vis_' % (str(args.f_type), str(args.attack)) + str(args.dataset) + '_' + str(args.network) + '_' + str(args.eps)

    check_dir(save_dir)

    inv = NetInversion(net, network=args.network)

    if args.f_type == 'posneg':
        sl, pu, nu = 4, 1, 2
        fig = inv.pos_neg_invert(selected_layer=sl, positive_unit=pu, negative_unit=nu)
        layer_info = [sl, pu, nu]

        save_img = network_vis(fig, layer_info, args.f_type)
        save_img.save(save_dir + '/L%d_P%d_N%d.png' % (sl, pu, nu))

    elif args.f_type == 'single':
        sl, su = 4, 1

        fig = inv.single_invert(selected_layer=sl, target_unit=su)
        layer_info = [sl, su]

        save_img = network_vis(fig, layer_info, args.f_type)
        save_img.save(save_dir + '/L%d_C%d.png' % (sl, su))

    elif args.f_type == 'combine':
        sl1, sl2, tu1, tu2 = 4, 4, 1, 2
        fig = inv.combine_invert(selected_layer1=sl1, selected_layer2=sl2, target_unit1=tu1, target_unit2=tu2)
        layer_info = [sl1, sl2, tu1, tu2]

        save_img = network_vis(fig, layer_info, args.f_type)
        save_img.save(save_dir + '/L%d_%d_C%d_%d.png' % (sl1, sl2, tu1, tu2))

    else:
        raise Exception(" [*] Wrong selection of visualization method .")

    print("\n [*] Inversion Img is saved")

if __name__ == '__main__':
    set_random(777)
    #net_visualize()
    visualizaition()






