import torch
checkpoint = torch.load('checkpoint/cifar10_awp_wide34_best.t7')
torch.save(checkpoint['net'], './awp_34.pth')