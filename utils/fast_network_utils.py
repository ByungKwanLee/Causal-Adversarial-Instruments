import torch
from models.vgg import vgg
from models.resnet import resnet
from models.wide import wide_resnet
from models.densenet import densenet
from models.causal_network import causal
from models.instrument_network import exogenous

def get_network(network, depth, dataset, ch=False):

    if dataset == 'cifar10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).cuda()
        std = torch.tensor([0.2023, 0.1994, 0.2010]).cuda()
    elif dataset == 'svhn': # later, it should be updated
        mean = torch.tensor([0.43090966, 0.4302428, 0.44634357]).cuda()
        std = torch.tensor([0.19759192, 0.20029082, 0.19811132]).cuda()
    elif dataset == 'cifar100':
        mean = torch.tensor([0.5071, 0.4867, 0.4408]).cuda()
        std = torch.tensor([0.2675, 0.2565, 0.2761]).cuda()
    elif dataset == 'tiny':
        mean = torch.tensor([0.48024578664982126, 0.44807218089384643, 0.3975477478649648]).cuda()
        std = torch.tensor([0.2769864069088257, 0.26906448510256, 0.282081906210584]).cuda()
    elif dataset == 'imagenet':
        mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
        std = torch.tensor([0.229, 0.224, 0.225]).cuda()

    if network == 'vgg':
        model = vgg(depth=depth, dataset=dataset, mean=mean, std=std)
    elif network == 'resnet':
        model = resnet(depth=depth, dataset=dataset, mean=mean, std=std)
    elif network == 'wide':
        model = wide_resnet(depth=depth, widen_factor=10, dataset=dataset, mean=mean, std=std)
    elif network == 'dense':
        model = densenet(depth=depth, dataset=dataset, mean=mean, std=std)
    elif network == 'causal':
        model = causal(dataset=dataset, ch=ch)
    elif network == 'instrument':
        model = exogenous(dataset=dataset, ch=ch)
    else:
        raise NotImplementedError

    return model
