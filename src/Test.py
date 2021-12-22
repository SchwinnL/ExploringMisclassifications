import torch
import torch.optim
import torch.nn.functional as F
import numpy as np
from .Adversarial import do_autoattack
from .Evaluation import save_result_dict
from .Utils import get_attack_key

def test(conf, model, test_loader, key, attack_conf):
    result_dict = {}
    if conf.que_attacks:
        result_dict[key] = "started"
    save_result_dict(conf, result_dict, name=conf.model.name + "_metrics")

    model.eval()

    total_acc = 0
    total_confidence = 0
    l2_norm = 0
    linf_norm = 0
    max_2_norm = 0
    max_linf_norm = 0

    pred_logits = np.empty((0, 10))
    n = 0
    print("Testing: {}".format(key))
    batch_init_key = ""
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        if i == 0 and "batch_tune" in attack_conf:
            attack_conf = do_batch_tune(conf, model, X, y, attack_conf)
            batch_init_key = get_attack_key(attack_conf["key"], conf, conf.dataset, conf.loss_function, attack_conf)
        delta = do_autoattack(conf, model, X, y, attack_conf)
        with torch.no_grad():
            pred = model(X + delta)
            total_acc += (pred.max(1)[1] == y).sum().item()
            total_confidence += torch.sum(F.softmax(pred, dim=1).max(1)[0])
            l2_norm += torch.sum(torch.norm(delta.view(delta.shape[0], -1), dim=1, p=2)).item()
            linf_norm += torch.sum(torch.norm(delta.view(delta.shape[0], -1), dim=1, p=float("inf"))).item()
            c_max_linf = torch.max(torch.norm(delta.view(delta.shape[0], -1), dim=1, p=float("inf")))
            c_max_l2 = torch.max(torch.norm(delta.view(delta.shape[0], -1), dim=1, p=2))
            if c_max_linf > max_linf_norm:
                max_linf_norm = c_max_linf
            if c_max_l2 > max_2_norm:
                max_2_norm = c_max_l2

            pred_logits = np.concatenate((pred_logits, pred.cpu().numpy()), 0)
            n += y.size(0)
    if batch_init_key != "":
        result_dict[key + " batch_init"] = batch_init_key
    result_dict[key + " l2_norm"] = l2_norm / n
    result_dict[key + " linf_norm"] = linf_norm / n
    result_dict[key + " max_l2_norm"] = max_2_norm
    result_dict[key + " max_linf_norm"] = max_linf_norm
    result_dict[key] = total_acc / n
    result_dict[key + " confidence"] = total_confidence.item() / n
    save_result_dict(conf, result_dict, name=conf.model.name + "_metrics")

def do_batch_tune(conf, model, X, y, attack_conf):
    best_values = {}
    best_acc = X.size(0)
    for key in attack_conf["batch_tune"]:
        current_attack_conf = attack_conf.copy()
        values = attack_conf["batch_tune"][key]
        best_values[key] = 0
        for value in values:
            current_attack_conf[key] = value
            delta = do_autoattack(conf, model, X, y, current_attack_conf)
            with torch.no_grad():
                pred = model(X + delta)
                current_acc = (pred.max(1)[1] == y).sum().item()
                if current_acc < best_acc:
                    best_values[key] = value
                    best_acc = current_acc
    new_attack_conf = attack_conf.copy()
    new_attack_conf.pop("batch_tune")
    for key in best_values:
        new_attack_conf[key] = best_values[key]
    return new_attack_conf