import sys
sys.path.append('.')
from model import SRL_BERT
from utils import get_dataloaders
from functions import train
import torch
import json


def train_SRL():
    train_loader, val_loader, test_loader, num_senses, num_roles = get_dataloaders("datasets/preprocessed/", batch_size=32, shuffle=True)

    tests = {
        "SRL_BERT_concat_100": {
            "model_name": "bert-base-uncased", # "bert-base-uncased", "bert-large-uncased"
            "combine_method": "concatenation",
            "role_layers": [100],
            "norm_layer": False,
            "dim_reduction": 0
        },
        "SRL_BERT_concat_100_norm": {
            "model_name": "bert-base-uncased", # "bert-base-uncased", "bert-large-uncased"
            "combine_method": "concatenation",
            "role_layers": [100],
            "norm_layer": True,
            "dim_reduction": 0
        },
        "SRL_BERT_mean_100": {
            "model_name": "bert-base-uncased", # "bert-base-uncased", "bert-large-uncased"
            "combine_method": "mean",
            "role_layers": [100],
            "norm_layer": False,
            "dim_reduction": 0
        },
        "SRL_BERT_mean_100_norm": {
            "model_name": "bert-base-uncased", # "bert-base-uncased", "bert-large-uncased"
            "combine_method": "mean",
            "role_layers": [100],
            "norm_layer": True,
            "dim_reduction": 0
        },
        "SRL_BERT_gated_100": {
            "model_name": "bert-base-uncased", # "bert-base-uncased", "bert-large-uncased"
            "combine_method": "gating",
            "role_layers": [100],
            "norm_layer": False,
            "dim_reduction": 0
        },
        "SRL_BERT_gated_100_norm": {
            "model_name": "bert-base-uncased", # "bert-base-uncased", "bert-large-uncased"
            "combine_method": "gating",
            "role_layers": [50],
            "norm_layer": True,
            "dim_reduction": 0
        },
        "SRL_BERT_gated_transform_100": {
            "model_name": "bert-base-uncased", # "bert-base-uncased", "bert-large-uncased"
            "combine_method": "gating_transform",
            "role_layers": [100],
            "norm_layer": False,
            "dim_reduction": 0
        },
        "SRL_BERT_gated_transform_100_norm": {
            "model_name": "bert-base-uncased", # "bert-base-uncased", "bert-large-uncased"
            "combine_method": "gating_transform",
            "role_layers": [100],
            "norm_layer": True,
            "dim_reduction": 0
        },
    }

    for test in tests:
        print(f"\nTraining model {test}")
        tests[test]["num_senses"] = num_senses
        tests[test]["num_roles"] = num_roles
        model = SRL_BERT(**tests[test])
        train(model, train_loader, val_loader, test_loader,
            epochs=10, init_lr=0.0001, scheduler_step=2, scheduler_gamma=0.9,
            device='cuda', name=test)
        # Save the model
        torch.save(model.state_dict(), f"models/{test}.pt")
        # Save json with the test parameters
        with open(f"models/{test}.json", "w") as f:
            json.dump(tests[test], f, indent=4)

        print("-"*50)

if __name__ == '__main__':
    train_SRL()
