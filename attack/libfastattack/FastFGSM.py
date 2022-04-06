import torch
from torch.cuda.amp import GradScaler, autocast
from torchattacks.attack import Attack


class FastFGSM(Attack):

    def __init__(self, model, eps=0.007):
        super().__init__("FastFGSM", model)
        self.eps = eps
        self._supported_mode = ['default', 'targeted']
        self.scaler = GradScaler()

    def forward(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_images = images.clone().detach()

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

        # Accelerating Gradient
        scaled_loss = self.scaler.scale(cost)
        # Update adversarial images
        grad = torch.autograd.grad(scaled_loss, adv_images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images_ = adv_images.detach() + self.eps*grad.sign()
        return torch.clamp(adv_images_, min=0, max=1).detach()