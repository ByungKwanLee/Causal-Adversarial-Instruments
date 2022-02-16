import torch
from models.vgg import vgg
from models.resnet import resnet
from models.wide import wide_resnet
from models.densenet import densenet


def get_network(network, depth, dataset, gpu):

    if dataset == 'cifar10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).to(gpu)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).to(gpu)
    elif dataset == 'svhn': # later, it should be updated
        mean = torch.tensor([0.43090966, 0.4302428, 0.44634357]).to(gpu)
        std = torch.tensor([0.19759192, 0.20029082, 0.19811132]).to(gpu)
    elif dataset == 'cifar100':
        mean = torch.tensor([0.5071, 0.4867, 0.4408]).to(gpu)
        std = torch.tensor([0.2675, 0.2565, 0.2761]).to(gpu)
    elif dataset == 'tiny':
        mean = torch.tensor([0.48024578664982126, 0.44807218089384643, 0.3975477478649648]).to(gpu)
        std = torch.tensor([0.2769864069088257, 0.26906448510256, 0.282081906210584]).to(gpu)
    elif dataset == 'imagenet':
        mean = torch.tensor([0.485, 0.456, 0.406]).to(gpu)
        std = torch.tensor([0.229, 0.224, 0.225]).to(gpu)

    if network == 'vgg':
        return vgg(depth=depth, dataset=dataset, mean=mean, std=std)
    elif network == 'resnet':
        return resnet(depth=depth, dataset=dataset, mean=mean, std=std)
    elif network == 'wide':
        return wide_resnet(depth=depth, widen_factor=10, dataset=dataset, mean=mean, std=std)
    elif network == 'dense':
        return densenet(depth=depth, dataset=dataset, mean=mean, std=std)
    else:
        raise NotImplementedError
