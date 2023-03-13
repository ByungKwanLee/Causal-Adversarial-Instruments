import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torchattacks.attack import Attack

class FastMIM(Attack):

    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, decay=1.0):
        super().__init__("FastMIM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
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

        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True

            with autocast():
                outputs = self.model(adv_images)

                # Calculate loss
                if self._targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels)

            scaled_loss = self.scaler.scale(cost)

            # Update adversarial images
            grad = torch.autograd.grad(scaled_loss, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images