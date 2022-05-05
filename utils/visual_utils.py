import torch
import warnings
import torch.backends.cudnn as cudnn
import scipy.ndimage as nd

from utils.utils import *
from torchvision import transforms

from lucent.optvis import render, param, transform, objectives
from lucent.modelzoo import inceptionv1
from lucent.misc.io.showing import animate_sequence
warnings.filterwarnings("ignore")

class SpInversion(object):
    def __init__(self, latent_z, model, dataset=None, fwd=False):
        super(SpInversion, self).__init__()
        #self.radn_img = torch.zeros((1, 3, 224,224)).to(device)
        self.model = model
        self.latent_z = latent_z
        self.cossim_pow = 2.0
        self.epochs = 512
        self.learning_rate = 5e-2
        self.fwd = fwd
        self.dataset = dataset

        if self.dataset == 'tiny':
            self.img_size = 64
            self.blur_constraint = -0.5
            self.cos_constant = 1e-6

        elif self.dataset == 'imagenet':
            self.img_size = 224
            self.blur_constraint = -0.5
            self.cos_constant = 1e-6

        else:
            self.img_size = 32
            self.blur_constraint = 0.01
            self.cos_constant = 1e-10


    @staticmethod
    def grad_on_off(model, switch=False):
        for param in model.parameters():
            param.requires_grad=switch

    @staticmethod
    def normalize(x):
        mean = x.mean(dim=(2,3), keepdim=True)
        std = x.std(dim=(2,3), keepdim=True)
        return (x-mean) / (std + 1e-10)

    def invert(self, image):
        r_transforms = [transform.pad(8, mode='constant', constant_value=.5),
                        transform.jitter(8),
                        transform.random_scale([0.9, 0.95, 1.05, 1.1] + [1] * 4),
                        transform.random_rotate(list(range(-5, 5)) + [0] * 5),
                        transform.jitter(2),
                        Resize((self.img_size, self.img_size))]

        self.model.eval()

        if self.fwd:
            images = image.cuda()
            ref_acts = self.model(images, pop=True).detach()
        else:
            ref_acts = self.latent_z.detach()

        params, image_f = param.image(self.img_size)
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)

        for i in range(self.epochs):
            acts = self.model(transform.compose(r_transforms)(image_f()), pop=True)

            dot = (acts * ref_acts).sum()
            mag = torch.sqrt(torch.sum(ref_acts ** 2))
            cossim = dot / (self.cos_constant + mag)
            cos_loss = - dot * cossim ** self.cossim_pow

            with torch.no_grad():
                t_input_blurred = torch_blur(image_f())
            blur_loss = self.blur_constraint * torch.sum((image_f() - t_input_blurred) ** 2)

            tot_loss = cos_loss + blur_loss

            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step()

        return tensor_to_img_array(image_f())

class NetInversion(object):
    def __init__(self, model, network):
        super(NetInversion, self).__init__()
        self.model = model
        self.network = network

    def pos_neg_invert(self, selected_layer, positive_unit, negative_unit):
        param_f = lambda: param.image(224, batch=2)
        if self.network == 'vgg':
            s_layer = 'vgg_features_%d'%(selected_layer)

        else:
            raise Exception(" [*] Wrong network description.")

        obj = objectives.channel(s_layer, positive_unit, batch=1) - objectives.channel(s_layer, negative_unit, batch=0)
        fig = render.render_vis(self.model, obj, param_f, show_inline=True)

        return fig

    def single_invert(self, selected_layer, target_unit):
        if self.network == 'vgg':
            neuron1 = ('vgg_features_%d'%(selected_layer), target_unit)

        elif self.network == 'resnet':
            neuron1 = ('layer%d' % (selected_layer), target_unit)

        elif self.network == 'dense':
            neuron1 = ('vgg_features_%d' % (selected_layer), target_unit)

        elif self.network == 'wide':
            neuron1 = ('vgg_features_%d' % (selected_layer), target_unit)

        else:
            raise Exception(" [*] Wrong network description.")

        param_f = lambda: param.image(224, batch=1)
        C = lambda neuron1: objectives.channel(*neuron1, batch=0)

        fig = render.render_vis(self.model, C(neuron1), param_f, show_inline=True)

        return fig

    def combine_invert(self, selected_layer1, selected_layer2, target_unit1, target_unit2):
        if self.network == 'vgg':
            neuron1 = ('vgg_features_%d' % (selected_layer1), target_unit1)
            neuron2 = ('vgg_features_%d' % (selected_layer2), target_unit2)

        elif self.network == 'resnet':
            neuron1 = ('layer%d' % (selected_layer1), target_unit1)
            neuron2 = ('layer%d' % (selected_layer2), target_unit2)

        elif self.network == 'dense':
            neuron1 = ('layer%d' % (selected_layer1), target_unit1)
            neuron2 = ('layer%d' % (selected_layer2), target_unit2)

        elif self.network == 'wide':
            neuron1 = ('layer%d' % (selected_layer1), target_unit1)
            neuron2 = ('layer%d' % (selected_layer2), target_unit2)

        else:
            raise Exception(" [*] Wrong network description.")

        param_f = lambda: param.image(224, batch=3)

        C = lambda neuron1, neuron2: objectives.channel(*neuron1, batch=0) + objectives.channel(*neuron2, batch=1) + \
                                     objectives.channel(*neuron1, batch=2) + objectives.channel(*neuron2, batch=2)

        fig = render.render_vis(self.model, C(neuron1, neuron2), param_f, show_inline=True)

        return fig
