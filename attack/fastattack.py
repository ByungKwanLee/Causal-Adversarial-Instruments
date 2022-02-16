from attack.libfastattack.FastPGD import FastPGD
from attack.libfastattack.FastFGSM import FastFGSM
from attack.libfastattack.FastFGSMTrain import FastFGSMTrain

def attack_loader(net, attack, eps, steps, dataset):

    # Gradient Clamping based Attack
    # torch attacks
    if attack == "fgsm":
        return FastFGSM(model=net, eps=eps)

    elif attack == "fgsm_train":
        return FastFGSMTrain(model=net, eps=eps)

    elif attack == "pgd":
        return FastPGD(model=net, eps=eps,
                                alpha=eps/steps*2.3, steps=steps, random_start=True)

