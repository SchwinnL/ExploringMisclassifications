import torch
import torch.nn as nn
from src.CustomEnums import ModelName
from src.models.ResNetMustafa import ResNetMustafa
from src.models.ResNetJinRinard import ResNetJinRinard
from src.models.ResNetZhang import WideResNetZhang
from src.Configuration import Conf
import robustbench as rb
from robustness.model_utils import make_and_restore_model
from robustness.datasets import ImageNet
import torchvision.models as models

def get_model(conf):
    model = None
    if conf.model == ModelName.Mustafa2019:
        model = ResNetMustafa()
    elif conf.model == ModelName.JinRinard2020:
        model = ResNetJinRinard()
    elif conf.model == ModelName.Zhang2021Geometry:
        model = WideResNetZhang()
    elif conf.model == ModelName.ImageNetRobustLibrary:
        dir = conf.model_save_path("", use_continue_training=False)
        model, _ = make_and_restore_model(arch="resnet50", dataset=ImageNet(Conf.get_imagenet_path()), resume_path=dir)
        model = RobustnessWraper(model)
    elif conf.model == ModelName.ImageNetFastIsBetter:
        model = models.__dict__["resnet50"]()
        checkpoint = torch.load(conf.model_save_path("Best"))
        state_dict = load_state_dict(checkpoint)
        model.load_state_dict(state_dict)
        model = ImageNetFastIsBetterWraper(model)
    else:
        dir = conf.model_save_path("", use_continue_training=False)
        model_name = conf.model.name.replace("_L2", "")
        dataset_name = conf.dataset.name

        if conf.dataset.name == "cifar10":
            dir = dir.replace("cifar10/" + conf.model_norm + "/" + conf.model.name.replace("_L2", "") + ".pt", "")
            model = rb.utils.load_model(model_name=model_name, dataset=dataset_name, model_dir=dir, norm=conf.model_norm)
        else:
            dir = dir.replace("cifar100/" + conf.model_norm + "/" + conf.model.name.replace("_L2", "") + ".pt", "")
            model = rb.utils.load_model(model_name=model_name, dataset=dataset_name, model_dir=dir, norm=conf.model_norm)
    model = model.to('cuda:0')
    return model

class ImageNetFastIsBetterWraper(nn.Module):
    def __init__(self, model):
        super(ImageNetFastIsBetterWraper, self).__init__()
        self.model = model
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()

    def forward(self, x):
        x = (x - self.mean) / self.std
        y = self.model(x)
        return y

class RobustnessWraper(nn.Module):
    def __init__(self, model):
        super(RobustnessWraper, self).__init__()
        self.model = model

    def forward(self, x):
        y, _ = self.model(x)
        return y

def load_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint["state_dict"]
        keys = list(checkpoint.keys())
        for key in keys:
            checkpoint[key.replace("module.", "")] = checkpoint[key]
            del checkpoint[key]
        return checkpoint