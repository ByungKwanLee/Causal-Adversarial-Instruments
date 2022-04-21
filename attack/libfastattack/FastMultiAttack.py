import torch
from torch.cuda.amp import autocast
from torchattacks.attack import Attack

class FastMultiAttack(Attack):

    def __init__(self, attacks, verbose=False):

        # Check validity
        ids = []
        for attack in attacks:
            ids.append(id(attack.model))

        if len(set(ids)) != 1:
            raise ValueError("At least one of attacks is referencing a different model.")

        super().__init__("MultiAttack", attack.model)
        self.attacks = attacks
        self.verbose = verbose
        self._multi_atk_records = [0.0]
        self._supported_mode = ['default']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        batch_size = images.shape[0]
        fails = torch.arange(batch_size).to(self.device)
        final_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        multi_atk_records = [batch_size]

        for _, attack in enumerate(self.attacks):
            adv_images = attack(images[fails], labels[fails])

            with autocast():
                outputs = self.model(adv_images)
            _, pre = torch.max(outputs.data, 1)

            corrects = (pre == labels[fails])
            wrongs = ~corrects

            succeeds = torch.masked_select(fails, wrongs)
            succeeds_of_fails = torch.masked_select(torch.arange(fails.shape[0]).to(self.device), wrongs)

            final_images[succeeds] = adv_images[succeeds_of_fails]

            fails = torch.masked_select(fails, corrects)
            multi_atk_records.append(len(fails))

            if len(fails) == 0:
                break

        return final_images

