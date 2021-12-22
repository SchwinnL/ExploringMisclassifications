import numpy as np
import  os
import torch

from global_config import train_keys, test_keys, optional, optional_test

class Conf:
    def __init__(self, kwargs_all):
        self.parse_train_params(train_keys + test_keys + optional + optional_test, **kwargs_all)
        if self.result_path == None:
            self.result_path = "Results"
        self.kwargs_name = self.__dict__.copy()
        for key in (test_keys + optional_test):
            self.kwargs_name.pop(key, None)
        self.save(self.kwargs_name)

    def parse_train_params(self, keys, **kwargs):
        allowed_keys = keys
        self.__dict__.update((key, None) for key in allowed_keys)
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)
        Conf.change_random_seed(self.seed)

    def full_model_name(self, short=True):
        description = self.get_full_description(short)
        fn = ""
        for d in description:
            fn += d
        return fn[:-1].replace(".", "-")

    def get_full_description(self, short=True):
        description = []
        def get_name(key, var):
            if hasattr(var, 'name'):
                if short:
                    return str(var.value)
                else:
                    return var.name
            elif var != None:
                return str(var)
            else:
                return ""

        for key in self.kwargs_name:
            var = self.kwargs_name[key]
            if var != None and var != []:
                if isinstance(var, dict):
                   var = list(var.values())
                if isinstance(var, list):
                    if len(var) != 0:
                        S = self.get_key_string(key, short) + "("
                        for val in var:
                            s = get_name(key, val)
                            if s != "":
                                S += s + "_"
                        S = S[:-1] + ")_"
                        description.append(S)
                else:
                    s = get_name(key, var)
                    if s != "":
                        description.append(self.get_key_string(key, short) + "(" + s + ")_")
        return description

    def get_key_string(self, key, short):
        if short:
            short = ""
            if "_" in key:
                split = key.split("_")
                for s in split:
                    short += s[0]
            else:
                short += key[0]
            return short
        else:
            return key

    def create_paths(self, paths):
        pn = ""
        for p in paths:
            pn += p
            if not os.path.isdir(pn):
                os.mkdir(pn)


    def save_path(self, dir, sub_dirs=[], create_path=True):
        paths = []
        paths.append(self.experiment_path)
        paths.append(dir + "/")

        if dir != "Models":
            paths.append("data_set(" + self.dataset.name + ")/")
        else:
            paths.append(self.dataset.name + "/")

        if len(sub_dirs) != 0:
            for dir in sub_dirs:
                pn = dir + "/"
                paths.append(pn)
        if create_path:
            self.create_paths(paths)

        if dir == "Models":
            paths.append(self.model_norm + "/" + self.model.name.replace("_L2", "") + ".pt")
        return "".join(paths)

    def confs_save_path(self, folder="Results"):
        return self.save_path(folder) + "configuration.txt"

    def tensorboard_log_path(self):
        return self.save_path("Tensorboard")

    def model_save_path(self, type, creat_path=True, use_continue_training=True):
        type = ""
        if self.continue_training and use_continue_training:
            type = self.continue_training["name"]

        dir = self.save_path("Models", create_path=creat_path)
        dir = dir.replace(".pt", type + ".pt")
        return dir

    def result_save_path(self, name=""):
        if name == "":
            name = "metrics"
        return self.save_path(self.result_path) + name + ".txt"

    def image_save_path(self, sub_dir = []):
        sub_dir.insert(0, "Images")
        path = self.save_path("Results", sub_dir)
        self.create_paths([path])
        return path

    def save(self, save_dict):
        old = {}
        if os.path.isfile(self.confs_save_path()):
            old = self.load_conf_dict()
        file = open(self.confs_save_path(), 'w')
        old.update(save_dict)
        configuration_text = self.get_string_representation(old)
        file.write(configuration_text)
        file.close()

    def load_conf_dict(self):
        dict = {}
        file = open(self.confs_save_path(), 'r').read()
        if file != "":
            lines = file.strip().split("\n")
            for line in lines:
                split = line.split(": ")
                if len(split) == 2:
                    key = split[0]
                    value = split[1]
                    dict[key] = value
        return dict

    def get_string_representation(self, info_dict):
        variables = ""
        if not (info_dict is None):
            for key in info_dict:
                variables += key + ": " + str(info_dict[key]) + "\n"
        return variables

    @staticmethod
    def change_random_seed(new_seed):
        new_seed = (new_seed + 1) * 42
        Conf.random_seed = new_seed
        np.random.seed(new_seed)
        torch.manual_seed(new_seed)
        torch.cuda.manual_seed(new_seed)





