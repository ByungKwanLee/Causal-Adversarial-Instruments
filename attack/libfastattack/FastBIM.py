import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torchattacks.attack import Attack

class FastBIM(Attack):

    def __init__(self, model, eps=4/255, alpha=1/255, steps=0):
        super().__init__("FastBIM", model)
        self.eps = eps
        self.alpha = alpha
        if steps == 0:
            self.steps = int(min(eps*255 + 4, 1.25*eps*255))
        else:
            self.steps = steps
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

        ori_images = images.clone().detach()

        for _ in range(self.steps):
            images.requires_grad = True

            # Accelerating forward propagation
            with autocast():
                outputs = self.model(images)

                # Calculate loss
                if self._targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels)

            # Update adversarial images with gradient scaler applied
            scaled_loss = self.scaler.scale(cost)

            # Update adversarial images
            grad = torch.autograd.grad(scaled_loss, images,
                                       retain_graph=False,
                                       create_graph=False)[0]

            adv_images = images + self.alpha*grad.sign()
            a = torch.clamp(ori_images - self.eps, min=0)
            b = (adv_images >= a).float()*adv_images \
                + (adv_images < a).float()*a
            c = (b > ori_images+self.eps).float()*(ori_images+self.eps) \
                + (b <= ori_images + self.eps).float()*b
            images = torch.clamp(c, max=1).detach()

        return images