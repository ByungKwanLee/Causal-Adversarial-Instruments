import os
import torch
import torchvision
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage.transform import resize
import io
from PIL import Image, ImageDraw, ImageFont

selectedFont = ImageFont.truetype(os.path.join('usr/share/fonts/', 'NanumGothic.ttf'), size=15)

def do_freeze(net):
    for params in net.parameters():
        params.requires_grad = False

def rprint(str, rank):
    if rank==0:
        print(str)

def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)

def checkpoint_module(checkpoint, net):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def Resize(size):
    def inner(image_t):
        return F.interpolate(image_t, size=size, mode='bilinear')

    return inner

def get_pseudo(adv_output):
    idx = adv_output.argmax(dim=1)
    b, c = adv_output.shape
    psuedo_onehot = torch.FloatTensor(b, c).cuda()
    psuedo_onehot.zero_()
    psuedo_onehot.scatter_(1, idx.unsqueeze(-1), 1)

    return psuedo_onehot

def get_onehot(adv_output, targets):
    b, c = adv_output.shape
    psuedo_onehot = torch.FloatTensor(b, c).cuda()
    psuedo_onehot.zero_()
    psuedo_onehot.scatter_(1, targets.unsqueeze(-1), 1)

    return psuedo_onehot

def show_with_var(pred_buf, dataset):
    if dataset == 'imagenet':
        object_categories = np.linspace(0, 999, num=1000)
    elif dataset == 'cifar10' or 'svhn':
        object_categories = np.linspace(0, 9, num=10)
    elif dataset == 'cifar100':
        object_categories = np.linspace(0, 99, num=100)

    pred = pred_buf / pred_buf[5]


    """Display single mnist digit next to the variance per class"""
    fig, axs = plt.subplots(2, 3, figsize=(70,15))
    axs[0][0].bar(object_categories, pred[5].cpu().detach().numpy(), color='green')
    axs[0][0].set_ylim([0, 4])
    axs[1][0].bar(object_categories, pred[0].cpu().detach().numpy(), color='green')
    axs[1][0].set_ylim([0, 4])

    axs[0][1].bar(object_categories, pred[1].cpu().detach().numpy(), color='red')
    axs[0][1].set_ylim([0, 4])
    axs[1][1].bar(object_categories, pred[2].cpu().detach().numpy(), color='blue')
    axs[1][1].set_ylim([0, 4])

    axs[0][2].bar(object_categories, pred[3].cpu().detach().numpy(), color='blue')
    axs[0][2].set_ylim([0, 4])
    axs[1][2].bar(object_categories, pred[4].cpu().detach().numpy(), color='blue')
    axs[1][2].set_ylim([0, 4])

    axs[0][0].title.set_text('True Label')
    axs[1][0].title.set_text('Clean Label')

    axs[0][1].title.set_text('Adv Label')
    axs[1][1].title.set_text('Inst Label')

    axs[0][2].title.set_text('Treat Label')
    axs[1][2].title.set_text('Causal Label')

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)

    return img

def torch_blur(tensor, out_c=3, ):
    depth = tensor.shape[1]
    weight = np.zeros([depth, depth, out_c, out_c])
    for ch in range(depth):
        weight_ch = weight[ch, ch, :, :]
        weight_ch[ :  ,  :  ] = 0.5
        weight_ch[1:-1, 1:-1] = 1.0
    weight_t = torch.tensor(weight).float().cuda()
    conv_f = lambda t: F.conv2d(t, weight_t, None, 1, 1)

    return conv_f(tensor) / conv_f(torch.ones_like(tensor))

def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image

def network_vis(fig, layer_info, f_type):
    bg_img = Image.new("RGB", (224 * len(fig[0]) + 20 * (len(fig[0]) + 1), 224 + 40), color=(255, 255, 255))

    for i in range(len(fig[0])):
        img = Image.fromarray((fig[0][i] * 255).astype(np.uint8))

        bg_img.paste(img, (20 * (i + 1) + 224 * i, 20))

    if f_type == 'posneg':
        draw = ImageDraw.Draw(bg_img)
        draw.text((20, 0), 'Positive: L%2d / C%3d' % (layer_info[0], layer_info[1]), fill='blue', font=selectedFont)
        draw.text((224 + 40, 0), 'Negative: L%2d / C%3d' % (layer_info[0], layer_info[2]), fill='red', font=selectedFont)

    elif f_type == 'single':
        draw = ImageDraw.Draw(bg_img)
        draw.text((20, 0), 'Single_L%2d_%3d' % (layer_info[0], layer_info[1]), fill='blue', font=selectedFont)

    elif f_type == 'combine':
        draw = ImageDraw.Draw(bg_img)
        draw.text((20, 0), 'L%2d_%3d' % (layer_info[0], layer_info[2]), fill='blue', font=selectedFont)
        draw.text((224 + 40, 0), 'L%2d_%3d' % (layer_info[1], layer_info[3]), fill='blue', font=selectedFont)
        draw.text((224 * 2 + 60, 0), 'Combined', fill='red', font=selectedFont)

    else:
        raise Exception("Wrong argument of FV selection")

    return bg_img

def causal_vis(img, inv, label, dataset=None):
    img = np.transpose(img.squeeze().cpu().detach().numpy(), [1, 2, 0])
    ori_img = Image.fromarray((img * 255).astype(np.uint8))
    ori_adv = Image.fromarray((inv[0] * 255).astype(np.uint8))
    ori_causal = Image.fromarray((inv[1] * 255).astype(np.uint8))
    ori_treat = Image.fromarray((inv[2] * 255).astype(np.uint8))
    ori_inst = Image.fromarray((inv[3] * 255).astype(np.uint8))

    ori_img = ori_img.resize((224, 224), Image.NEAREST)
    ori_adv = ori_adv.resize((224, 224), Image.NEAREST)
    ori_causal = ori_causal.resize((224, 224), Image.NEAREST)
    ori_treat = ori_treat.resize((224, 224), Image.NEAREST)
    ori_inst = ori_inst.resize((224, 224), Image.NEAREST)

    bg_img = Image.new("RGB", (224 * 5 + 20 * 6, 224 * 1 + 40), color=(255, 255, 255))

    bg_img.paste(ori_img, (20, 20))
    bg_img.paste(ori_adv, (224 * 1 + 20 * 2, 20))
    bg_img.paste(ori_causal, (224 * 2 + 20 * 3, 20))
    bg_img.paste(ori_treat, (224 * 3 + 20 * 4, 20))
    bg_img.paste(ori_inst, (224 * 4 + 20 * 5, 20))

    if dataset == 'svhn':
        o_label = [str(i) for i in range(10)]
    elif dataset == 'cifar10':
        o_label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset == 'tiny':
        o_label = list(range(1, 201))
    elif dataset == 'imagenet':
        f = open("./utils/imagenet_label.txt", 'r')
        lines = f.readlines()
        target = lines[label[0]].split(',')[0].split(':')[1].split("'")[1]
        pred1 = lines[label[1]].split(',')[0].split(':')[1].split("'")[1]
        pred2 = lines[label[2]].split(',')[0].split(':')[1].split("'")[1]
        pred3 = lines[label[3]].split(',')[0].split(':')[1].split("'")[1]
        pred4 = lines[label[4]].split(',')[0].split(':')[1].split("'")[1]
        o_label = [target, pred1, pred2, pred3, pred4]
        label = [0, 1, 2, 3, 4]
        f.close()

    draw = ImageDraw.Draw(bg_img)

    draw.text((20, 0), 'Image: ' + str(o_label[label[0]]), fill='blue', font=selectedFont)
    draw.text((224 * 1 + 20 * 2, 0), 'ADV Inv: ' + str(o_label[label[1]]), fill='red', font=selectedFont)
    draw.text((224 * 2 + 20 * 3, 0), 'Causal Inv: ' + str(o_label[label[2]]), fill='red', font=selectedFont)
    draw.text((224 * 3 + 20 * 4, 0), 'Treat Inv: ' + str(o_label[label[3]]), fill='red', font=selectedFont)
    draw.text((224 * 4 + 20 * 5, 0), 'Inst Inv: ' + str(o_label[label[4]]), fill='red', font=selectedFont)

    return bg_img

def feature_vis(img, inv, label, dataset=None):
    img = np.transpose(img.squeeze().cpu().detach().numpy(), [1, 2, 0])

    ori_img = Image.fromarray((img * 255).astype(np.uint8))
    ori_inv = Image.fromarray((inv * 255).astype(np.uint8))
    ori_img = ori_img.resize((224, 224), Image.NEAREST)
    ori_inv = ori_inv.resize((224, 224), Image.NEAREST)

    bg_img = Image.new("RGB", (224 * 2 + 20 * 3, 224 * 1 + 40), color=(255, 255, 255))

    bg_img.paste(ori_img, (20, 20))
    bg_img.paste(ori_inv, (224 * 1 + 20 * 2, 20))

    if dataset == 'svhn':
        o_label = [str(i) for i in range(10)]
    elif dataset == 'cifar10':
        o_label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset == 'tiny':
        o_label = list(range(1, 201))
    elif dataset == 'imagenet':
        f = open("./utils/imagenet_label.txt", 'r')
        lines = f.readlines()
        target = lines[label[0]].split(',')[0].split(':')[1].split("'")[1]
        pred = lines[label[1]].split(',')[0].split(':')[1].split("'")[1]
        o_label = [target, pred]
        f.close()

    draw = ImageDraw.Draw(bg_img)

    draw.text((20, 0), 'Image: ' + str(o_label[label[0]]), fill='blue', font=selectedFont)
    draw.text((224 * 1 + 20 * 2, 0), 'Inversion: ' + str(o_label[label[1]]), fill='red', font=selectedFont)

    return bg_img

def get_resolution(epoch, min_res, max_res, end_ramp, start_ramp):
    assert min_res <= max_res

    if epoch <= start_ramp:
        return min_res

    if epoch >= end_ramp:
        return max_res

    # otherwise, linearly interpolate to the nearest multiple of 32
    interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
    final_res = int(np.round(interp[0] / 32)) * 32

    return final_res

def get_randomresizedcrop(epoch, min_res, max_res, end_ramp, start_ramp):
    assert min_res <= max_res

    if epoch <= start_ramp:
        return torchvision.transforms.Resize((min_res, min_res))

    if epoch >= end_ramp:
        return torchvision.transforms.Resize((max_res, max_res))

    # otherwise, linearly interpolate to the nearest multiple of 32
    interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
    final_res = int(np.round(interp[0] / 32)) * 32

    return torchvision.transforms.RandomResizedCrop((final_res, final_res))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        assert False

def imshow(img, norm=False):
    img = img.cpu().numpy()
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

def print_configuration(args, rank):
    dict = vars(args)
    if rank == 0:
        print('------------------Configurations------------------')
        for key in dict.keys():
            print("{}: {}".format(key, dict[key]))
        print('-------------------------------------------------')

def KLDivergence(q, p):
    kld = q * (q / p).log()
    return kld.sum(dim=1)

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


# attack loader
from attack.fastattack import attack_loader
from tqdm import tqdm
def test_robustness(net, testloader, criterion, attack_list, rank):
    net.eval()
    test_loss = 0

    attack_module = {}
    for attack_name in attack_list:
        attack_module[attack_name] = attack_loader(net=net, attack=attack_name, eps=0.03, steps=30) \
                                                                                if attack_name != 'plain' else None

    for key in attack_module:
        total = 0
        correct = 0
        prog_bar = tqdm(enumerate(testloader), total=len(testloader), leave=False)
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.cuda(), targets.cuda()
            if key != 'plain':
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
            if (key == 'auto') or (key == 'fab'):
                if batch_idx >= int(len(testloader) * 0.3):
                    break

        rprint(f'{key}: {100. * correct / total}%', rank)
