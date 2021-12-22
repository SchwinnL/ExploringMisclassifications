from src.CustomEnums import *

train_keys = ['model', 'dataset',
              'batch_size', 'seed', "current_plot"]
test_keys = ["experiment_path", "data_path", 'model_norm',
             'attacks', 'que_attacks', 'test_size']
optional = ["seed", "continue_training"]
optional_test = ["result_path"]

all_enums = {ModelName, DataSetName, AdversarialAttacks, Norm}

enums = {"model", "dataset", "norm_in", "norm_out", "attacks"}
