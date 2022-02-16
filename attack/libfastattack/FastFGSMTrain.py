import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torchattacks.attack import Attack


class FastFGSMTrain(Attack):

    def __init__(self, model, eps=0.007):
        super().__init__("FastFGSMTrain", model)
        self.eps = eps
        self._supported_mode = ['default', 'targeted']
        self.scaler = GradScaler()

    def forward(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_images = images.clone().detach()

        # Starting at a uniformly random point
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = torch.nn.CrossEntropyLoss()

        adv_images.requires_grad = True

        # Accelarating forward propagation
        with autocast():
            outputs = self.model(adv_images)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

        # Accerlating Gradient
        scaled_loss = self.scaler.scale(cost)
        # Update adversarial images
        grad = torch.autograd.grad(scaled_loss, adv_images,
                                   retain_graph=False, create_graph=False)[0]
        grad /= scaled_loss / cost

        adv_images_ = adv_images.detach() + 1.25 * self.eps*grad.sign()
        delta = torch.clamp(adv_images_ - images, min=-self.eps, max=self.eps)
        return torch.clamp(images + delta, min=0, max=1).detach()