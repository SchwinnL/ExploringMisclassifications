from src.CustomEnums import ModelName
import torch

def is_string(val):
    try:
        str(val)
        return True
    except:
        return False

def is_float(val, print_val=False):
    try:
        float(val)
        return True
    except:
        if print_val:
            print(val)
        return False

def get_attack_key(key, conf, dataset, attack_conf=None):
    if not (attack_conf is None):
        for dict_key in attack_conf:
            if dict_key != "key" and dict_key != "type" and dict_key != "name":
                key += "_" + str(attack_conf[dict_key])
    if not (dataset is None) and conf.dataset != dataset:
        key += "_" + dataset.name
    return key.replace(":", "")

def load_model(conf, model):
    if conf.model == ModelName.Mustafa2019 or conf.model == ModelName.JinRinard2020 or conf.model == ModelName.Zhang2021Geometry:
        checkpoint = torch.load(conf.model_save_path("Best"))

        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint = checkpoint["state_dict"]
            keys = list(checkpoint.keys())
            for key in keys:
                checkpoint[key.replace("module.", "")] = checkpoint[key]
                del checkpoint[key]
        model.load_state_dict(checkpoint)
        model = model.to('cuda:0')
    return model