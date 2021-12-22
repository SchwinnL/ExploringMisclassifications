from src.Configuration import Conf
from src.Datasets import get_data_set
from src.models.Models import get_model
from src.Evaluation import contains_result
from src.Utils import get_attack_key, load_model
from src.Test import test
import itertools
from read_yaml import open_yaml
import argparse

experiment_path = "./Experiments/"
data_path = "./Data/"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_path", default=experiment_path)
    parser.add_argument("--data_path", default=data_path)
    parser.add_argument("--yaml_file", default=["CIFAR10"], nargs='*')
    return parser.parse_args()

def run_experiment(conf):
    test_size = conf.test_size
    train_loader, valid_loader, test_loader = get_data_set(conf, test_size=test_size)
    model = get_model(conf)
    result_dict = {}
    result_dict["Model"] = conf.model.name
    result_dict["Model_Path"] = conf.model_save_path("Best")

    model = load_model(conf, model)

    dataset = conf.dataset
    for attack in conf.attacks:
        if test_size < 1:
            attack["Test_Size"] = "ts" + str(test_size)
        key = get_attack_key(attack["key"], conf, dataset, attack_conf=attack)
        if not contains_result(conf, key):
            test(conf, model, test_loader, key, attack)

input_args = get_args()
for yaml_file in input_args.yaml_file:
    print("Yaml File:", yaml_file)
    arguments, keys = open_yaml(yaml_file)
    runs = list(itertools.product(*arguments))
    for run in runs:
        current_args = {}
        current_args["experiment_path"] = input_args.experiment_path
        current_args["data_path"] = input_args.data_path
        for i in range(len(keys)):
            current_args[keys[i]] = run[i]
        conf = Conf(current_args)
        run_experiment(conf)