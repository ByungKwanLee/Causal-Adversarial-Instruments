import os
import torchvision
import numpy as np
import matplotlib.pyplot as plt

def get_resolution(epoch, min_res, max_res, end_ramp, start_ramp):
    assert min_res <= max_res

    if epoch <= start_ramp:
        return torchvision.transforms.Resize((min_res, min_res))

    if epoch >= end_ramp:
        return torchvision.transforms.Resize((max_res, max_res))

    # otherwise, linearly interpolate to the nearest multiple of 32
    interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
    final_res = int(np.round(interp[0] / 32)) * 32

    return torchvision.transforms.Resize((final_res, final_res))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        assert False

def imshow(img, norm=False):
    img = img[0].cpu().numpy()
    plt.imshow(np.transpose(np.array(img / 255 if norm else img, dtype=np.float32), (1, 2, 0)))
    plt.show()

def pl(a):
    plt.plot(a.cpu())
    plt.show()

def sc(a):
    plt.scatter(range(len(a.cpu())), a.cpu(), s=2, color='darkred', alpha=0.5)
    plt.show()

def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

def print_configuration(args):
    dict = vars(args)
    print('------------------Configurations------------------')
    for key in dict.keys():
        print("{}: {}".format(key, dict[key]))
    print('-------------------------------------------------')

def KLDivergence(q, p):
    kld = q * (q / p).log()
    return kld.sum(dim=1)

from pprint import pprint

class StairCaseLRScheduler(object):
    def __init__(self, start_at, interval, decay_rate):
        self.start_at = start_at
        self.interval = interval
        self.decay_rate = decay_rate

    def __call__(self, optimizer, iteration):
        start_at = self.start_at
        interval = self.interval
        decay_rate = self.decay_rate
        if (start_at >= 0) \
                and (iteration >= start_at) \
                and (iteration + 1) % interval == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_rate
                print('[%d]Decay lr to %f' % (iteration, param_group['lr']))

    @staticmethod
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            return lr


class PresetLRScheduler(object):
    """Using a manually designed learning rate schedule rules.
    """
    def __init__(self, decay_schedule):
        # decay_schedule is a dictionary
        # which is for specifying iteration -> lr
        self.decay_schedule = decay_schedule
        print('=> Using a preset learning rate schedule:')
        pprint(decay_schedule)
        self.for_once = True

    def __call__(self, optimizer, iteration):
        for param_group in optimizer.param_groups:
            lr = self.decay_schedule.get(iteration, param_group['lr'])
            param_group['lr'] = lr

    @staticmethod
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            return lr