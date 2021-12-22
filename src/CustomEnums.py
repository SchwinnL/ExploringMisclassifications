from enum import Enum

class SaveType(Enum):
    result = 0
    image = 1
    tensorboard = 2
    configurations = 3

    def __str__(self):
        return self.name

class ModelName(Enum):
    Wong2020Fast = 1
    Sehwag2020Hydra = 2
    Wang2020Improving = 3
    Hendrycks2019Using = 4
    Rice2020Overfitting = 5
    Zhang2019Theoretically = 6
    Engstrom2019Robustness = 7
    Chen2020Adversarial = 8
    Gowal2020Uncovering_28_10_extra = 9
    Huang2020Self = 10
    Pang2020Boosting = 11
    Carmon2019Unlabeled = 12
    Ding2020MMA = 13
    Zhang2019You = 14
    Zhang2020Attacks = 15
    Wu2020Adversarial_extra = 16
    Wu2020Adversarial = 17
    Augustin2020Adversarial_L2 = 18
    Engstrom2019Robustness_L2 = 19
    Rice2020Overfitting_L2 = 20
    Rony2019Decoupling_L2 = 21
    Ding2020MMA_L2 = 22
    Wu2020Adversarial_L2 = 23
    Mustafa2019 = 25
    JinRinard2020 = 26
    Gowal2020Uncovering_34_20 = 28
    Rebuffi2021Fixing_R18_ddpm = 30
    Gowal2020Uncovering = 31
    Zhang2021Geometry = 32
    Sehwag2021Proxy_R18 = 33
    Chen2020Efficient = 34
    Cui2020Learnable_34_10_LBGAT0 = 35
    Sitawarin2020Improving = 36
    Rade2021Helper_R18_ddpm = 37
    Cui2020Learnable_34_10_LBGAT6 = 38
    ImageNetRobustLibrary = 39
    ImageNetFastIsBetter = 40

    def __str__(self):
        return self.name

class DataSetName(Enum):
    cifar10 = 0
    cifar100 = 1
    imagenet = 2

    def __str__(self):
        return self.name

class AdversarialAttacks(Enum):
    apgd = 0
    def __str__(self):
        return self.name

class Norm(Enum):
    l2 = 0
    linf = 1