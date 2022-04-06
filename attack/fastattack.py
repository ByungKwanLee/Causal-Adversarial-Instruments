from attack.libfastattack.FastPGD import FastPGD
from attack.libfastattack.FastFGSM import FastFGSM
from attack.libfastattack.FastFGSMTrain import FastFGSMTrain
from attack.libfastattack.FastTPGD import FastTPGD
from attack.libfastattack.FastCWLinf import FastCWLinf
from attack.libfastattack.FastAPGD import FastAPGD
from attack.libfastattack.FastBIM import FastBIM
from attack.libfastattack.FastAutoAttack import FastAutoAttack

def attack_loader(net, attack, eps, steps):

    # Gradient Clamping based Attack
    # torch attacks
    if attack == "fgsm":
        return FastFGSM(model=net, eps=eps)

    elif attack == "fgsm_train":
        return FastFGSMTrain(model=net, eps=eps)

    elif attack == "pgd":
        return FastPGD(model=net, eps=eps,
                                alpha=eps/steps*2.3, steps=steps, random_start=True)

    elif attack == "kld_pgd":
        return FastTPGD(model=net, eps=eps, alpha=eps/steps*2.3, steps=steps)

    elif attack == "cw_linf":
        return FastCWLinf(model=net, c=0.1, lr=0.1, steps=200)

    elif attack == "apgd":
        return FastAPGD(model=net, eps=eps, loss='ce', steps=30)

    elif attack == "bim":
        return FastBIM(model=net, eps=eps, alpha=1/255)

    elif attack == "auto":
        return FastAutoAttack(model=net, eps=eps)