# Future
from __future__ import print_function

# warning ignore
import warnings
warnings.filterwarnings("ignore")

import argparse

from utils.fast_network_utils import get_network
from utils.fast_data_utils import get_fast_dataloader
from utils.utils import *

# fetch args
parser = argparse.ArgumentParser()

# model parameter
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--network', default='vgg', type=str)
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--base', default='adv', type=str)
parser.add_argument('--batch_size', default=64, type=float)
parser.add_argument('--gpu', default='2', type=str) # necessarily one gpu id!!!!
args = parser.parse_args()

def test_number_of_correct_and_wrong(net, c_net, testloader, attack_list, eps, inv_causal, rank):
    KL = lambda x, y: (x.softmax(dim=1) * (x.softmax(dim=1).log() - y.softmax(dim=1).log())).sum(dim=1).mean()

    net.eval()
    c_net.eval()

    attack_module = {}
    for attack_name in attack_list:
        attack_module[attack_name] = attack_loader(net=net, attack=attack_name, eps=eps, steps=30)

    for key in attack_module:

        case_1 = 0
        case_2 = 0
        case_3 = 0
        case_4 = 0

        case_5 = 0
        case_6 = 0
        case_7 = 0
        case_8 = 0

        prog_bar = tqdm(enumerate(testloader), total=len(testloader), leave=False)
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.cuda(), targets.cuda()
            adv_inputs = attack_module[key](inputs, targets)

            with autocast():
                # clean feature
                clean_feature = net(inputs, pop=True)
                clean_outputs = net(clean_feature.clone(), int=True)
                clean_targets = (clean_outputs.max(1)[1] == targets)

                # adv feature
                adv_feature = net(adv_inputs, pop=True)
                adv_outputs = net(adv_feature.clone(), int=True)
                adv_targets = (adv_outputs.max(1)[1] == targets)

                # causal feature and output
                causal_feature = clean_feature + c_net(adv_feature - clean_feature)
                causal_outputs = net(causal_feature.clone(), int=True)
                causal_targets = (causal_outputs.max(1)[1] == targets)

                # inv causal feature
                inv_inputs = inv_causal(inputs, targets, causal_outputs.detach())
                inv_feature = net(inv_inputs, pop=True)
                inv_outputs = net(inv_feature.clone(), int=True)
                inv_targets = (inv_outputs.max(1)[1] == targets)


            # CASE
            case_1 += ((clean_targets == True) * (adv_targets == True)).sum().item()
            case_2 += ((clean_targets == False) * (adv_targets == True)).sum().item()
            case_3 += ((clean_targets == True) * (adv_targets == False)).sum().item()
            case_4 += ((clean_targets == False) * (adv_targets == False)).sum().item()

            # CASE
            case_5 += ((clean_targets == True) * (inv_targets == True)).sum().item()
            case_6 += ((clean_targets == False) * (inv_targets == True)).sum().item()
            case_7 += ((clean_targets == True) * (inv_targets == False)).sum().item()
            case_8 += ((clean_targets == False) * (inv_targets == False)).sum().item()

            desc = ('[Inv/%s] Case 1 : %.2f%% | Case 2 : %.2f%% | Case 3 : %.2f%% | Case 4 : %.2f%% | Case 5 : %.2f%% | Case 6 : %.2f%% | Case 7 : %.2f%% | Case 8 : %.2f%%'
                    % (key, case_1 / (batch_idx+1), case_2 / (batch_idx+1), case_3 / (batch_idx+1), case_4 / (batch_idx+1),
                       case_5 / (batch_idx+1), case_6 / (batch_idx+1), case_7 / (batch_idx+1), case_8 / (batch_idx+1)))
            prog_bar.set_description(desc, refresh=True)


        rprint('------------------CASE------------------', rank)
        rprint(f'{key}: Case 1 -> {case_1 / (batch_idx+1):.3f}', rank)
        rprint(f'{key}: Case 2 -> {case_2 / (batch_idx+1):.3f}', rank)
        rprint(f'{key}: Case 3 -> {case_3 / (batch_idx+1):.3f}', rank)
        rprint(f'{key}: Case 4 -> {case_4 / (batch_idx+1):.3f}', rank)
        rprint(f'{key}: Case 5 -> {case_5 / (batch_idx+1):.3f}', rank)
        rprint(f'{key}: Case 6 -> {case_6 / (batch_idx+1):.3f}', rank)
        rprint(f'{key}: Case 7 -> {case_7 / (batch_idx+1):.3f}', rank)
        rprint(f'{key}: Case 8 -> {case_8 / (batch_idx+1):.3f}', rank)




def main_worker():

    # print configuration
    print_configuration(args, 0)


    torch.cuda.set_device(f'cuda:{args.gpu}')

    # init model
    net = get_network(network=args.network,
                      depth=args.depth,
                      dataset=args.dataset).cuda()
    net.eval()

    # init causal net
    c_net = get_network(network='causal', depth=None, dataset=args.dataset,
                        ch=True if args.network=='wide' else False).cuda()
    c_net.eval()


    # init dataloader
    _, testloader, _ = get_fast_dataloader(dataset=args.dataset,
                                        train_batch_size=1,
                                        test_batch_size=args.batch_size)

    # Loading checkpoint
    checkpoint_name = 'checkpoint/pretrain/%s/%s_%s_%s%s_best.t7' % (
    args.dataset, args.dataset, args.base, args.network, args.depth)

    checkpoint = torch.load(checkpoint_name, map_location=torch.device(torch.cuda.current_device()))
    checkpoint_module(checkpoint['net'], net)
    rprint("This test : {}".format(checkpoint_name), 0)


    # Load Causal Network
    checkpoint_name = 'checkpoint/causal/%s/%s_causal_%s%s_best.t7' % (args.dataset, args.dataset, args.network, args.depth)
    checkpoint = torch.load(checkpoint_name, map_location=torch.device(torch.cuda.current_device()))
    checkpoint_module(checkpoint['c_net'], c_net)
    rprint("This test : {}".format(checkpoint_name), 0)


    # inv causal
    inv_causal = attack_loader(net=net, attack='causalpgd', eps=4/255 if args.dataset == 'tiny' else 0.03, steps=10)

    # test inversion
    test_inversion(net, c_net, testloader, attack_list=['pgd'], eps=4/255 if args.dataset == 'tiny' else 0.03, inv_causal=inv_causal, rank=0)
    # test_number_of_correct_and_wrong(net, c_net, testloader, attack_list=['pgd'], eps=4/255 if args.dataset == 'tiny' else 0.03, inv_causal=inv_causal, rank=0)

if __name__ == '__main__':
    main_worker()






