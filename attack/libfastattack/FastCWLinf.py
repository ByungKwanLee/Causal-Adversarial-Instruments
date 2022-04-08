import torch
from torch.cuda.amp import GradScaler, autocast
from torchattacks.attack import Attack

class FastCWLinf(Attack):

    def __init__(self, model, eps, scale, kappa=0, steps=1000):
        super().__init__("FastCWLinf", model)
        self.eps = eps
        self.alpha = eps/steps * 3
        self.kappa = kappa
        self.steps = steps
        self.scale = scale
        self._supported_mode = ['default', 'targeted']
        self.scaler = GradScaler()

    def forward(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_images = images.clone().detach()

        # Starting at a uniformly random point
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for step in range(self.steps):

            adv_images.requires_grad = True

            # Accelerating forward propagation
            with autocast():
                outputs = self.model(adv_images)
                cost = self.scale * self.f(outputs, labels).sum()

            # Update adversarial images with gradient scaler applied
            scaled_loss = self.scaler.scale(cost)

            grad = torch.autograd.grad(scaled_loss, adv_images, retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() - self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    def tanh_space(self, x):
        return 1/2*(torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        return self.atanh(x*2-1)

    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())

        if self._targeted:
            return torch.clamp((i-j), min=-self.kappa)
        else:
            return torch.clamp((j-i), min=-self.kappa)