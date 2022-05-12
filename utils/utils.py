import os
import torch
import torch.nn as nn
import torchvision
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import OrderedDict
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
    elif dataset == 'cifar10':
        object_categories = np.linspace(0, 9, num=10)
    elif dataset == 'svhn':
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

def inst_vis(img, inv, save_dir, batch_idx):
    img = np.transpose(img.squeeze().cpu().detach().numpy(), [1, 2, 0])
    ori_img = Image.fromarray((img * 255).astype(np.uint8))
    ori_cln = Image.fromarray((inv[0] * 255).astype(np.uint8))
    ori_adv = Image.fromarray((inv[1] * 255).astype(np.uint8))
    ori_causal = Image.fromarray((inv[2] * 255).astype(np.uint8))
    ori_treat = Image.fromarray((inv[3] * 255).astype(np.uint8))
    ori_inst = Image.fromarray((inv[4] * 255).astype(np.uint8))

    ori_img = ori_img.resize((224, 224), Image.NEAREST)
    ori_cln = ori_cln.resize((224, 224), Image.NEAREST)
    ori_adv = ori_adv.resize((224, 224), Image.NEAREST)
    ori_causal = ori_causal.resize((224, 224), Image.NEAREST)
    ori_treat = ori_treat.resize((224, 224), Image.NEAREST)
    ori_inst = ori_inst.resize((224, 224), Image.NEAREST)

    ori_img.save(save_dir + '/ori_img%d.png' % (batch_idx))
    ori_cln.save(save_dir + '/cln_img%d.png' % (batch_idx))
    ori_adv.save(save_dir + '/adv_img%d.png' % (batch_idx))
    ori_causal.save(save_dir + '/causal_img%d.png' % (batch_idx))
    ori_treat.save(save_dir + '/treat_img%d.png' % (batch_idx))
    ori_inst.save(save_dir + '/inst_img%d.png' % (batch_idx))


def causal_vis(img, inv, label, dataset=None):
    img = np.transpose(img.squeeze().cpu().detach().numpy(), [1, 2, 0])
    ori_img = Image.fromarray((img * 255).astype(np.uint8))
    ori_cln = Image.fromarray((inv[0] * 255).astype(np.uint8))
    ori_adv = Image.fromarray((inv[1] * 255).astype(np.uint8))
    ori_causal = Image.fromarray((inv[2] * 255).astype(np.uint8))
    ori_treat = Image.fromarray((inv[3] * 255).astype(np.uint8))
    ori_inst = Image.fromarray((inv[4] * 255).astype(np.uint8))

    ori_img = ori_img.resize((224, 224), Image.NEAREST)
    ori_cln = ori_cln.resize((224, 224), Image.NEAREST)
    ori_adv = ori_adv.resize((224, 224), Image.NEAREST)
    ori_causal = ori_causal.resize((224, 224), Image.NEAREST)
    ori_treat = ori_treat.resize((224, 224), Image.NEAREST)
    ori_inst = ori_inst.resize((224, 224), Image.NEAREST)

    bg_img = Image.new("RGB", (224 * 6 + 20 * 7, 224 * 1 + 40), color=(255, 255, 255))

    bg_img.paste(ori_img, (20, 20))
    bg_img.paste(ori_cln, (224 * 1 + 20 * 2, 20))
    bg_img.paste(ori_adv, (224 * 2 + 20 * 3, 20))
    bg_img.paste(ori_causal, (224 * 3 + 20 * 4, 20))
    bg_img.paste(ori_treat, (224 * 4 + 20 * 5, 20))
    bg_img.paste(ori_inst, (224 * 5 + 20 * 6, 20))

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
        pred5 = lines[label[5]].split(',')[0].split(':')[1].split("'")[1]
        o_label = [target, pred1, pred2, pred3, pred4, pred5]
        label = [0, 1, 2, 3, 4, 5]
        f.close()

    draw = ImageDraw.Draw(bg_img)

    draw.text((20, 0), 'Image: ' + str(o_label[label[0]]), fill='blue', font=selectedFont)
    draw.text((224 * 1 + 20 * 2, 0), 'Clean Inv: ' + str(o_label[label[1]]), fill='red', font=selectedFont)
    draw.text((224 * 2 + 20 * 3, 0), 'Adv Inv: ' + str(o_label[label[2]]), fill='red', font=selectedFont)
    draw.text((224 * 3 + 20 * 4, 0), 'Causal Inv: ' + str(o_label[label[3]]), fill='red', font=selectedFont)
    draw.text((224 * 4 + 20 * 5, 0), 'Treat Inv: ' + str(o_label[label[4]]), fill='red', font=selectedFont)
    draw.text((224 * 5 + 20 * 6, 0), 'Inst Inv: ' + str(o_label[label[5]]), fill='red', font=selectedFont)

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
        label = [0, 1]
        f.close()

    draw = ImageDraw.Draw(bg_img)

    draw.text((20, 0), 'Image: ' + str(o_label[label[0]]), fill='blue', font=selectedFont)
    draw.text((224 * 1 + 20 * 2, 0), 'Inversion: ' + str(o_label[label[1]]), fill='red', font=selectedFont)

    return bg_img


class SmoothCrossEntropyLoss(torch.nn.Module):
    """
    Soft cross entropy loss with label smoothing.
    """
    def __init__(self, smoothing=0.0, reduction='mean'):
        super(SmoothCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, input, target):
        num_classes = input.shape[1]
        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes)
        target = (1. - self.smoothing) * target + self.smoothing / num_classes
        logprobs = torch.nn.functional.log_softmax(input, dim=1)
        loss = - (target * logprobs).sum(dim=1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


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

# attack loader
from attack.fastattack import attack_loader
from tqdm import tqdm
from torch.cuda.amp import autocast
def test_whitebox(net, testloader, attack_list, eps, rank):
    net.eval()

    attack_module = {}
    for attack_name in attack_list:
        attack_module[attack_name] = attack_loader(net=net, attack=attack_name, eps=eps, steps=30) \
                                                                                if attack_name != 'plain' else None
    for key in attack_module:
        total = 0
        correct = 0
        prog_bar = tqdm(enumerate(testloader), total=len(testloader), leave=False)
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.cuda(), targets.cuda()
            if key != 'plain':
                inputs = attack_module[key](inputs, targets)
            with autocast():
                outputs = net(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('[White-Box-Test/%s] Acc: %.2f%% (%d/%d)'
                    % (key, 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

        rprint(f'{key}: {100. * correct / total:.2f}%', rank)

def test_blackbox(plain_net, adv_net, testloader, attack_list, eps, rank):
    plain_net.eval()
    adv_net.eval()

    attack_module = {}
    for attack_name in attack_list:
        attack_module[attack_name] = attack_loader(net=plain_net, attack=attack_name, eps=eps, steps=30)

    for key in attack_module:
        total = 0
        correct = 0
        prog_bar = tqdm(enumerate(testloader), total=len(testloader), leave=False)
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs = attack_module[key](inputs, targets)

            with autocast():
                outputs = adv_net(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('[Black-Box-Test/%s] Acc: %.2f%% (%d/%d)'
                    % (key, 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

        rprint(f'{key}: {100. * correct / total:.2f}%', rank)


def test_inversion(net, c_net, testloader, attack_list, eps, inv_causal, rank):
    KL = lambda x, y: (x.softmax(dim=1) * (x.softmax(dim=1).log() - y.softmax(dim=1).log())).sum(dim=1).mean()
    net.eval()
    c_net.eval()

    eps_list = [eps]

    attack_module = {}
    for attack_name in attack_list:
        attack_module[attack_name] = []
        for epsilon in eps_list:
            attack_module[attack_name].append(attack_loader(net=net, attack=attack_name, eps=epsilon, steps=10))

    for key in attack_module:
        for idx, attack in enumerate(attack_module[key]):

            kl_inv = 0
            kl_clean = 0
            kl_adv = 0

            prog_bar = tqdm(enumerate(testloader), total=len(testloader), leave=False)
            for batch_idx, (inputs, targets) in prog_bar:
                inputs, targets = inputs.cuda(), targets.cuda()
                adv_inputs = attack(inputs, targets)

                with autocast():
                    # clean feature
                    clean_feature = net(inputs, pop=True)
                    clean_outputs = net(clean_feature.clone(), int=True)

                    # adv feature
                    adv_feature = net(adv_inputs, pop=True)
                    adv_outputs = net(adv_feature.clone(), int=True)

                    # causal feature and output
                    causal_feature = clean_feature + c_net(adv_feature - clean_feature)
                    causal_outputs = net(causal_feature.clone(), int=True)



                # inv causal feature
                inv_inputs = inv_causal(inputs, targets, causal_outputs.detach())
                with autocast():
                    inv_outputs = net(inv_inputs)

                    # KLD
                    kl_inv += KL(inv_outputs, causal_outputs).item()
                    kl_clean += KL(clean_outputs, causal_outputs).item()
                    kl_adv += KL(adv_outputs, causal_outputs).item()



                desc = ('[%s] KLD (Inv: %.3f, Clean: %.3f, Adv: %.3f)'
                        % (key, 10**3*kl_inv/(batch_idx+1), 10**3*kl_clean/(batch_idx+1), 10**3*kl_adv/(batch_idx+1)))
                prog_bar.set_description(desc, refresh=True)

            rprint('------------------KLD------------------', rank)
            rprint(f'{key}/{eps_list[idx]:.3f}: Inv -> {10**3*kl_inv/(batch_idx+1):.3f}', rank)
            rprint(f'{key}/{eps_list[idx]:.3f}: Clean -> {10**3*kl_clean/(batch_idx+1):.3f}', rank)
            rprint(f'{key}/{eps_list[idx]:.3f}: Adv -> {10**3*kl_adv/(batch_idx+1):.3f}', rank)
            rprint('---------------------------------------', rank)

def inv_eps(dataset, network):

    if dataset == 'cifar10' and network=='vgg':
        inv_eps = 0.03

    elif dataset == 'cifar10' and network == 'resnet':
        inv_eps = 4/255

    elif dataset == 'cifar10' and network == 'wide':
        inv_eps = 2/255

    elif dataset == 'svhn' and network=='vgg':
        inv_eps = 4/255

    elif dataset == 'svhn' and network=='resnet':
        inv_eps = 1/255

    elif dataset == 'svhn' and network=='wide':
        inv_eps = 1/255

    elif dataset == 'tiny' and network=='vgg':
        inv_eps = 1/255

    elif dataset == 'tiny' and network == 'resnet':
        inv_eps = 0.5/255

    elif dataset == 'tiny' and network == 'wide':
        inv_eps = 0.5/255

    print(f'Data: {dataset}, Net: {network}, InvEps: {inv_eps*255:.1f}')
    return inv_eps


class MixAttack(object):
    def __init__(self, net, slowattack, fastattack, train_iters):
        self.net = net
        self.slowattack = slowattack
        self.fastattack = fastattack
        self.train_iters = train_iters
        self.ratio = 0.3
        self.current_iter = 0

    def __call__(self, inputs, targets):
        # training
        if self.net.training:
            adv_inputs = self.slowattack(inputs, targets) \
                if self._iter < self.train_iters * self.ratio else self.fastattack(inputs, targets)
            self.iter()
            self.check()
        # testing
        else:
            adv_inputs = self.fastattack(inputs, targets)
        return adv_inputs

    def iter(self):
        self.current_iter = self.current_iter+1

    def check(self):
        if self.train_iters == self.current_iter:
            self.current_iter = 0

    @property
    def _iter(self):
        return self.current_iter

# awp package
EPS = 1E-20
def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])

class AdvWeightPerturb(object):
    def __init__(self, model, proxy, lr, gamma, autocast, GradScaler):
        super(AdvWeightPerturb, self).__init__()
        self.model = model
        self.proxy = proxy
        self.gamma = gamma
        self.proxy_optim = torch.optim.SGD(proxy.parameters(), lr=lr)
        self.autocast = autocast
        self.scaler = GradScaler()

    def calc_awp(self, adv_inputs, targets):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        self.proxy_optim.zero_grad()
        with self.autocast():
            loss = -F.cross_entropy(self.proxy(adv_inputs), targets)

        # Accelerating backward propagation
        self.scaler.scale(loss).backward()
        self.scaler.step(self.proxy_optim)
        self.scaler.update()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)


# Causal Package
def causal_loss(logits_adv, logits_inv):
    KL = lambda x, y: (x.softmax(dim=1) * (x.softmax(dim=1).log() - y.softmax(dim=1).log())).sum(dim=1)
    return (KL(logits_adv, logits_inv)).mean()
