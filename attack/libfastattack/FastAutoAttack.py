import time
import torch.nn as nn
from torchattacks.attack import Attack
from attack.libfastattack.FastAPGD import FastAPGD
from attack.libfastattack.FastMultiAttack import FastMultiAttack


class FastAutoAttack(Attack):
    r"""
    AutoAttack in the paper 'Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks'
    [https://arxiv.org/abs/2003.01690]
    [https://github.com/fra31/auto-attack]
    Distance Measure : Linf, L2
    Arguments:
        model (nn.Module): model to attack.
        norm (str) : Lp-norm to minimize. ['Linf', 'L2'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: 0.3)
        version (bool): version. ['standard', 'plus', 'rand'] (Default: 'standard')
        n_classes (int): number of classes. (Default: 10)
        seed (int): random seed for the starting point. (Default: 0)
        verbose (bool): print progress. (Default: False)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    """
    def __init__(self, model, norm='Linf', eps=.3, seed=None, verbose=False):
        super().__init__("FastAutoAttack", model)
        self.norm = norm
        self.eps = eps
        self.seed = seed
        self.verbose = verbose
        self._supported_mode = ['default']

        self.autoattack = FastMultiAttack([
            FastAPGD(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, loss='ce', eot_iter=1, n_restarts=1),
            FastAPGD(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, loss='dlr', eot_iter=1, n_restarts=1),
        ])


    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = self.autoattack(images, labels)

        return adv_images

    def get_seed(self):
        return time.time() if self.seed is None else self.seed