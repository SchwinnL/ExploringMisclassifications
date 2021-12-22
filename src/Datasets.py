from .Configuration import Conf
from .CustomEnums import DataSetName
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

import os

def get_data_set(conf, test_size=1):
    dataset = conf.dataset
    batch_size = conf.batch_size
    train, valid, test, train_loader, valid_loader, test_loader = [None] * 6
    if dataset == DataSetName.cifar10:
        train, test = get_cifar10(conf)
    elif dataset == DataSetName.cifar100:
        train, test = get_cifar100(conf)
    elif dataset == DataSetName.imagenet:
        train, test = get_ImageNet(conf)
    else:
        print("Dataset not defined")
    train_loader, valid_loader, test_loader = train_valid_test_split(train, test, batch_size, train_split=0.9,
                                                                     test_size=test_size)
    return train_loader, valid_loader, test_loader

def get_cifar10(conf):
    transform_train = get_transform(DataSetName.cifar10, 3, 32, True)
    transforms_test = get_transform(DataSetName.cifar10, 3, 32, False)

    train = datasets.CIFAR10(os.path.join(conf.data_path, "CIFAR10-data"), train=True, download=True,
                                          transform=transform_train)
    test = datasets.CIFAR10(os.path.join(conf.data_path, "CIFAR10-data"), train=False, download=True,
                                         transform=transforms_test)
    return train, test

def get_cifar100(conf):
    transform_train = get_transform(DataSetName.cifar100, 3, 32, True)
    transforms_test = get_transform(DataSetName.cifar100, 3, 32, False)

    train = datasets.CIFAR100(conf.data_path + "/CIFAR100-data", train=True, download=True, transform=transform_train)
    test = datasets.CIFAR100(conf.data_path + "/CIFAR100-data", train=False, download=True, transform=transforms_test)
    return train, test

def get_ImageNet(conf):
    transform_train = get_transform(DataSetName.imagenet, 3, None, True)
    transforms_test = get_transform(DataSetName.imagenet, 3, None, False)

    train = datasets.ImageNet("IMAGENETPATH", split="train", transform=transform_train)
    test = datasets.ImageNet("IMAGENETPATH", split="val", transform=transforms_test)
    return train, test

def get_transform(data_set, channels, size, train):
    t = []
    # Horizontal flips and cropping for CIFAR10
    if train and (data_set == DataSetName.cifar10):
        t.append(transforms.Pad(4, padding_mode='reflect'))
        t.append(transforms.RandomHorizontalFlip())
        t.append(transforms.RandomCrop(size, padding=4))
    if data_set == DataSetName.imagenet:
        t.append(transforms.Resize(256))
        t.append(transforms.CenterCrop(224))
    t.append(transforms.ToTensor())
    transform = transforms.Compose(t)
    return transform

def get_mean_std(data_set):
    mean = 0.
    std = 1.
    # Standatarize for CIFAR10
    if data_set == DataSetName.cifar_10:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif data_set == DataSetName.cifar_100:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif data_set == DataSetName.imagenet:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    channels = get_dataset_information(data_set)[1]
    if channels == 3:
        return torch.tensor(mean).view(3, 1, 1).cuda(), torch.tensor(std).view(3, 1, 1).cuda()
    else:
        return torch.tensor(mean), torch.tensor(std)

def get_lower_and_upper_limits():
    lower_limit = 0.
    upper_limit = 1.
    return lower_limit, upper_limit

def train_valid_test_split(train, test, batch_size, train_split=0.9, test_size=1):
    total_count = len(train)
    train_count = int(train_split * total_count)
    val_count = total_count - train_count
    train, val = torch.utils.data.random_split(train, [train_count, val_count])

    if test_size != 1:
        test_count = int(len(test) * test_size)
        _count = len(test) - test_count
        test, _ = torch.utils.data.random_split(test, [test_count, _count])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(val, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, valid_loader, test_loader

def get_dataset_information(dataset):
    if dataset == DataSetName.mnist:
        return 10, 1, (1, 1, 28, 28)
    elif dataset == DataSetName.cifar10:
        return 10, 3, (1, 3, 32, 32)
    elif dataset == DataSetName.cifar_100:
        return 100, 3, (1, 3, 32, 32)

def get_data_from_loader(conf, loader):
    if conf.dataset == DataSetName.cifar10:
        data = loader.dataset.data.transpose(0, 3, 1, 2) / 255
        labels = loader.dataset.targets
    if conf.dataset == DataSetName.cifar100:
        data = loader.dataset.data.transpose(0, 3, 1, 2) / 255
        labels = loader.dataset.targets
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)