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
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--network', default='vgg', type=str)

parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--gpu', default='0', type=str)

parser.add_argument('--base', default='causal', type=str)
parser.add_argument('--batch_size', default=128, type=float)

# attack parameter
parser.add_argument('--attack', default='pgd', type=str)
parser.add_argument('--eps', default=0.03, type=float)
parser.add_argument('--steps', default=30, type=int)

# visualization parameter
parser.add_argument('--f_type', default='combine', type=str, help='option: posneg / single / combine')

args = parser.parse_args()

# GPU configurations
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

# init dataloader
_, testloader, _ = get_fast_dataloader(dataset=args.dataset, train_batch_size=1, test_batch_size=args.batch_size, dist=False)

# init model
net = get_network(network=args.network, depth=args.depth, dataset=args.dataset)
c_net = get_network(network='causal', depth=None, dataset=args.dataset)
z_net = get_network(network='instrument', depth=args.depth, dataset=args.dataset)

net = net.cuda()
c_net = c_net.cuda()
z_net = z_net.cuda()

# Load Plain Network
print('==> Loading Plain checkpoint..')
assert os.path.isdir('checkpoint/pretrain'), 'Error: no checkpoint directory found!'

# Loading checkpoint
net_checkpoint_name = 'checkpoint/pretrain/%s/%s_adv_%s%s_best.t7' % (args.dataset, args.dataset, args.network, args.depth)
causal_checkpoint_name = 'checkpoint/pretrain/%s/%s_causal_F_reg50_%s%s_best.t7' % (args.dataset, args.dataset, args.network, args.depth)

net_checkpoint = torch.load(net_checkpoint_name, map_location=lambda storage, loc: storage.cuda())['net']
c_net_checkpoint = torch.load(causal_checkpoint_name, map_location=lambda storage, loc: storage.cuda())['c_net']
z_net_checkpoint = torch.load(causal_checkpoint_name, map_location=lambda storage, loc: storage.cuda())['z_net']

print(" [*] Loading Checkpoints ...")
checkpoint_module(net_checkpoint, net)
checkpoint_module(c_net_checkpoint, c_net)
checkpoint_module(z_net_checkpoint, z_net)

# init criterion
criterion = nn.CrossEntropyLoss()

def test():
    net.eval()
    c_net.eval()
    z_net.eval()

    attack_module = {}
    # for attack_name in ['Plain', 'fgsm', 'pgd', 'cw_Linf', 'apgd', 'auto']:
    for attack_name in ['pgd']:
        args.attack = attack_name
        attack_module[attack_name] = attack_loader(net=net, attack=attack_name,
                                                   eps=args.eps, steps=args.steps,
                                                   )

    for key in attack_module:
        total = 0
        adv_correct, causal_correct, inst_correct, treat_correct = 0, 0 ,0, 0
        prog_bar = tqdm(enumerate(testloader), total=len(testloader), leave=True)

        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.cuda(), targets.cuda()
            adv_inputs = attack_module[key](inputs, targets)

            adv_feature = net(adv_inputs, pop=True)
            cln_feature = net(inputs, pop=True)
            residual = adv_feature - cln_feature

            adv_output = net(adv_feature, int=True)

            inst_feature = z_net(residual)
            inst_output = net(cln_feature + inst_feature, int=True)

            treat_feature = cln_feature + c_net(residual)
            treat_output = net(treat_feature, int=True)

            causal_feature = c_net(inst_feature)
            causal_output = net(cln_feature + causal_feature, int=True)

            _, adv_predicted = adv_output.max(1)
            _, inst_predicted = inst_output.max(1)
            _, treat_predicted = treat_output.max(1)
            _, causal_predicted = causal_output.max(1)

            total += targets.size(0)
            adv_correct += adv_predicted.eq(targets).sum().item()
            inst_correct += inst_predicted.eq(targets).sum().item()
            treat_correct += treat_predicted.eq(targets).sum().item()
            causal_correct += causal_predicted.eq(targets).sum().item()

            desc = ('[Test/%s] Adv: %.2f%% | Inst: %.2f%%| Treat: %.2f%% | Causal: %.2f%%'
                    % (key, 100. * adv_correct / total, 100. * inst_correct / total, 100. * treat_correct / total, 100. * causal_correct / total))
            prog_bar.set_description(desc, refresh=True)

            # fast eval
            if (key == 'apgd') or (key == 'auto') or (key == 'cw_Linf') or (key == 'cw'):
                if batch_idx >= int(len(testloader) * 0.3):
                    break

    #     attack_score.append(100. * correct / total)
    #
    # print('\n----------------Summary----------------')
    # print(args.steps, ' steps attack')
    # for key, score in zip(attack_module, attack_score):
    #     print(str(key), ' : ', str(score) + '(%)')
    # print('---------------------------------------\n')

def visualizaition():
    net.eval()
    c_net.eval()
    z_net.eval()

    if args.base == 'plain':
        save_dir = './results/feature_vis/clean_vis_' + str(args.dataset) + '_' + str(args.network)
    elif args.base == 'adv':
        save_dir = './results/feature_vis/%s_vis_' % (str(args.attack)) + str(args.dataset) + '_' + str(args.network) + '_' + str(args.eps)
    else:
        save_dir = './results/feature_vis/causal_t_vis_' + str(args.dataset) + '_' + str(args.network)

    check_dir(save_dir)
    attack = attack_loader(net=net, attack='pgd', eps=args.eps, steps=args.steps)

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        adv_inputs = attack(inputs, targets)

        adv_feature = net(adv_inputs, pop=True)
        cln_feature = net(inputs, pop=True)
        residual = adv_feature - cln_feature
        worst_feature = z_net(residual)

        inst_feature = cln_feature + worst_feature
        treat_feature = cln_feature + c_net(residual)
        causal_feature = cln_feature + c_net(worst_feature)

        _, adv_pred = net(adv_feature, int=True).max(1)
        _, causal_pred = net(causal_feature, int=True).max(1)
        _, treat_pred = net(treat_feature, int=True).max(1)
        _, inst_pred = net(inst_feature, int=True).max(1)

        if targets.item() != adv_pred.item():
            adv_inv = SpInversion(adv_feature.clone(), net, dataset=args.dataset).invert(inputs).squeeze()
            causal_inv = SpInversion(causal_feature.clone(), net, dataset=args.dataset).invert(inputs).squeeze()
            treat_inv = SpInversion(treat_feature.clone(), net, dataset=args.dataset).invert(inputs).squeeze()
            inst_inv = SpInversion(inst_feature.clone(), net, dataset=args.dataset).invert(inputs).squeeze()

            label = [targets.item(), adv_pred.item(), causal_pred.item(), treat_pred.item(), inst_pred.item()]
            inv = [adv_inv, causal_inv, treat_inv, inst_inv]

            save_img = causal_vis(inputs, inv, label, dataset=args.dataset)
            save_img.save(save_dir + '/inv_img%d.png' % (batch_idx))
            print("\n [*] Inversion Img%d is saved" % (batch_idx))

def class_prediction():
    net.eval()
    c_net.eval()
    z_net.eval()
    attack_module = {}

    for attack_name in ['pgd']:
        args.attack = attack_name
        attack_module[attack_name] = attack_loader(net=net, attack=attack_name,
                                                   eps=args.eps, steps=args.steps,)

    for key in attack_module:
        total = 0
        adv_correct, causal_correct, inst_correct, treat_correct = 0, 0, 0, 0
        prog_bar = tqdm(enumerate(testloader), total=len(testloader), leave=True)

        if args.dataset == 'imagnet':
            pred_buf = torch.zeros(6, 1000).cuda()
        elif args.dataset == 'cifar10' or 'svhn':
            pred_buf = torch.zeros(6, 10).cuda()
        elif args.dataset == 'cifar100':
            pred_buf = torch.zeros(6, 100).cuda()
        elif args.dataset == 'tiny':
            pred_buf = torch.zeros(6, 200).cuda()
        else:
            raise Exception("Worng Dataset")

        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.cuda(), targets.cuda()
            adv_inputs = attack_module[key](inputs, targets)

            adv_feature = net(adv_inputs, pop=True)
            cln_feature = net(inputs, pop=True)
            residual = adv_feature - cln_feature

            adv_output = net(adv_feature, int=True)
            cln_output = net(cln_feature, int=True)

            inst_feature = z_net(residual)
            inst_output = net(cln_feature + inst_feature, int=True)

            treat_feature = cln_feature + c_net(residual)
            treat_output = net(treat_feature, int=True)

            causal_feature = c_net(inst_feature)
            causal_output = net(cln_feature + causal_feature, int=True)

            pred_buf[0] += get_pseudo(cln_output).sum(0)
            pred_buf[1] += get_pseudo(adv_output).sum(0)
            pred_buf[2] += get_pseudo(inst_output).sum(0)
            pred_buf[3] += get_pseudo(treat_output).sum(0)
            pred_buf[4] += get_pseudo(causal_output).sum(0)
            pred_buf[5] += get_onehot(adv_output, targets).sum(0)

            _, adv_predicted = adv_output.max(1)
            _, inst_predicted = inst_output.max(1)
            _, treat_predicted = treat_output.max(1)
            _, causal_predicted = causal_output.max(1)

            total += targets.size(0)
            adv_correct += adv_predicted.eq(targets).sum().item()
            inst_correct += inst_predicted.eq(targets).sum().item()
            treat_correct += treat_predicted.eq(targets).sum().item()
            causal_correct += causal_predicted.eq(targets).sum().item()

            desc = ('[Test/%s] Adv: %.2f%% | Inst: %.2f%%| Treat: %.2f%% | Causal: %.2f%%'
                    % (key, 100. * adv_correct / total, 100. * inst_correct / total, 100. * treat_correct / total,
                       100. * causal_correct / total))
            prog_bar.set_description(desc, refresh=True)

        img = show_with_var(pred_buf, args.dataset)
        img.save('stat_%s.png' %(str(causal_checkpoint_name.split('/')[-1].split('.')[0])))

def net_visualize():
    net.eval()
    c_net.eval()
    z_net.eval()

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
    #visualizaition()
    class_prediction()
    #test()






