import sys
sys.path.append('.')
from model import SRL_BERT
from utils import get_dataloaders
from functions import train
import torch
import json


def train_SRL():
    tests = {
        "SRL_BERT_gated_redboth100_100_norm_L2": {
            "model_name": "bert-base-uncased",
            "combine_method": "gating",
            "role_layers": [100],
            "norm_layer": True,
            "dim_reduction": 100,
            "relation_reduction": True
        },
        "SRL_BERT_L_gated_redboth100_100_norm_L2": {
            "model_name": "bert-large-uncased",
            "combine_method": "gating",
            "role_layers": [100],
            "norm_layer": True,
            "dim_reduction": 100,
            "relation_reduction": True
        },
        "SRL_BERT_C_gated_redboth100_100_norm_L2": {
            "model_name": "bert-base-cased",
            "combine_method": "gating",
            "role_layers": [100],
            "norm_layer": True,
            "dim_reduction": 100,
            "relation_reduction": True
        },
        "SRL_BERT_LC_gated_redboth100_100_norm_L2": {
            "model_name": "bert-large-cased",
            "combine_method": "gating",
            "role_layers": [100],
            "norm_layer": True,
            "dim_reduction": 100,
            "relation_reduction": True
        },
    }

    for test in tests:
        train_loader, val_loader, test_loader, num_senses, num_roles = get_dataloaders("datasets/preprocessed/", batch_size=64, shuffle=True, model_name=tests[test]["model_name"])
        print(f"\nTraining model {test}")
        tests[test]["sense_classes"] = num_senses
        tests[test]["role_classes"] = num_roles
        model = SRL_BERT(**tests[test])
        print(f"Number of parameters in the model: {sum(p[1].numel() for p in model.named_parameters() if 'bert' not in p[0])}")
        train(model, train_loader, val_loader, test_loader,
            epochs=10, init_lr=0.0001, scheduler_step=2, scheduler_gamma=0.9, l2_lambda=1e-5,
            device='cuda', name=test)
        # Save the model
        torch.save(model.state_dict(), f"models/{test}.pt")
        # Save json with the test parameters
        with open(f"models/{test}.json", "w") as f:
            json.dump(tests[test], f, indent=4)

        print("-"*50)

if __name__ == '__main__':
    train_SRL()
