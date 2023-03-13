import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torchattacks.attack import Attack

class FastPGD(Attack):

    def __init__(self, model, eps=0.3,
                 alpha=2/255, steps=40, random_start=True):
        super().__init__("FastPGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']
        self.scaler = GradScaler()

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True

            # Accelerating forward propagation
            with autocast():
                outputs = self.model(adv_images)

                # Calculate loss
                if self._targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels)

            # Update adversarial images with gradient scaler applied
            scaled_loss = self.scaler.scale(cost)
            grad = torch.autograd.grad(scaled_loss, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        return adv_images