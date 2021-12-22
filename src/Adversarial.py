from .CustomEnums import DataSetName, Norm
from src.autoattack.autopgd_pt import APGDAttack

def get_eps(conf):
    if conf.dataset == DataSetName.cifar10 or conf.dataset == DataSetName.cifar100:
        eps = 8. / 255.
    elif conf.dataset == DataSetName.imagenet:
        eps = 4. / 255
    else:
        raise NameError("No perturbation budget defined for this dataset")
    return eps

def get_alpha(conf):
    if conf.dataset == DataSetName.cifar10 or conf.dataset == DataSetName.cifar100:
        alpha = 2. / 255.
    elif conf.dataset == DataSetName.imagenet:
        alpha = 1. / 255
    else:
        raise NameError("No alpha defined for this dataset")
    return alpha

def do_autoattack(conf, model, X, y, attack_conf):
    attack_iters = get_value("iters", 100, attack_conf)
    restarts = get_value("restarts", 1, attack_conf)
    norm = get_value("norm", Norm.linf, attack_conf)
    eps = get_value("eps", get_eps(conf), attack_conf)
    loss = get_value("loss", "ce", attack_conf)
    eot = get_value("EOT", 1, attack_conf)
    rand_output_scale = get_value("sigma", False, attack_conf)
    early_stop = get_value("early_stop", False, attack_conf)

    if norm == Norm.linf:
        norm = "Linf"
    elif norm == Norm.l2:
        norm = "L2"

    attack = APGDAttack(model, n_iter=attack_iters, n_restarts=restarts, norm=norm, eps=eps, loss=loss, eot_iter=eot, sigma=rand_output_scale, early_stop=early_stop)
    _, x_adv = attack.perturb(X, y, cheap=True)
    delta = x_adv - X
    return delta

def get_value(key, value, conf):
    if key in conf:
        return conf[key]
    else:
        return value

