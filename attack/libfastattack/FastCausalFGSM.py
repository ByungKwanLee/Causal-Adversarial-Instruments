import torch
from torch.cuda.amp import GradScaler, autocast
from torchattacks.attack import Attack


class FastCausalFGSM(Attack):

    def __init__(self, model, eps):
        super().__init__("FastCausalFGSM", model)
        self.eps = eps
        self._supported_mode = ['default', 'targeted']
        self.scaler = GradScaler()
        self.alpha = 2.5

    def forward(self, images, labels, causal_outputs):

        images = images.clone().detach().to(self.device)

        adv_images = images.clone().detach()

        # Starting at a uniformly random point
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
        adv_images.requires_grad = True

        # Accelarating forward propagation
        with autocast():
            outputs = self.model(adv_images)

            # Calculate loss
            cost = (outputs.softmax(dim=1) * (outputs.softmax(dim=1).log() - causal_outputs.softmax(dim=1).log())).mean()

        # Accelerating Gradient
        scaled_loss = self.scaler.scale(cost)
        # Update adversarial images
        grad = torch.autograd.grad(scaled_loss, adv_images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images_ = adv_images.detach() + self.alpha * self.eps*grad.sign()
        delta = torch.clamp(adv_images_ - images, min=-self.eps, max=self.eps)
        return torch.clamp(images + delta, min=0, max=1).detach()