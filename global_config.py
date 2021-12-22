from src.CustomEnums import *

train_keys = ['model', 'loss_function', 'feature_dim', 'dataset',
              'batch_size', 'epochs', 'lr', 'optimizer', 'training_type',
              'lr_schedule', 'seed', "pretrained"]
test_keys = ["train", "load", "plot", "test", "experiment_path", "data_path",
             "perturbation_multipliers", 'test_noise', 'layer_type', 'model_norm',
             'attacks', 'que_attacks', 'test_size', 'create_csm', 'plots']
optional = ["seed", "pretrained", "continue_training"]
optional_test = ["result_path", "plot_model"]

all_enums = {ModelName, DataSetName, AdversarialAttacks, Norm}

enums = {"model", "dataset", "lr_schedule", "optimizer", "training_type", "loss_function", "norm_in", "norm_out", "attacks", 'plots'}
