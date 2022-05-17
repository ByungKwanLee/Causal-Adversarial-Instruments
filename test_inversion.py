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
parser.add_argument('--batch_size', default=256, type=float)
parser.add_argument('--gpu', default='0', type=str) # necessarily one gpu id!!!!
args = parser.parse_args()

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
    inv_causal = attack_loader(net=net, attack='causalpgd', eps=inv_eps(args.dataset, args.network), steps=3)

    # test inversion
    test_inversion(net, c_net, testloader, attack_list=['fgsm_train'], eps=4/255 if args.dataset == 'tiny' else 0.03, inv_causal=inv_causal, rank=0)

if __name__ == '__main__':
    main_worker()






