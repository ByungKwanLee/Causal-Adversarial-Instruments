from attack.libfastattack.FastFGSM import FastFGSM
from attack.libfastattack.FastFGSMTrain import FastFGSMTrain
from attack.libfastattack.FastPGD import FastPGD
from attack.libfastattack.FastCausalPGD import FastCausalPGD
from attack.libfastattack.FastCWLinf import FastCWLinf
from attack.libfastattack.FastAPGD import FastAPGD
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

    elif attack == "causalpgd":
        return FastCausalPGD(model=net, eps=eps,
                                alpha=eps/steps*2.3, steps=steps, random_start=True)

    elif attack == "cw_linf":
        return FastCWLinf(model=net, eps=eps, scale=0.1, steps=100)

    elif attack == "ap":
        return FastAPGD(model=net, eps=eps, scale=0.1, loss='ce', steps=steps)

    elif attack == "dlr":
        return FastAPGD(model=net, eps=eps, scale=0.1, loss='dlr', steps=steps, n_restarts=2)

    elif attack == "aa":
        return FastAutoAttack(model=net, eps=eps)



