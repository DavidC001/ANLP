import sys
sys.path.append('.')
from model import SRL_BERT
from utils import get_dataloaders
from functions import train
import torch


def train_SRL():
    train_loader, val_loader, test_loader, num_senses, num_roles = get_dataloaders("datasets/preprocessed/", batch_size=64, shuffle=True)

    tests = {
        "SRL_BERT_concat_100": {
            "combine_method": "concat",
            "layers": [100],
            "norm_layer": False
        },
        "SRL_BERT_concat_100_norm": {
            "combine_method": "concat",
            "layers": [100],
            "norm_layer": True
        },
        "SRL_BERT_mean_100": {
            "combine_method": "mean",
            "layers": [100],
            "norm_layer": False
        },
        "SRL_BERT_mean_100_norm": {
            "combine_method": "mean",
            "layers": [100],
            "norm_layer": True
        },
        "SRL_BERT_gated_100": {
            "combine_method": "gated",
            "layers": [100],
            "norm_layer": False
        },
        "SRL_BERT_gated_100_norm": {
            "combine_method": "gated",
            "layers": [100],
            "norm_layer": True
        },
        "SRL_BERT_gated_transform_100": {
            "combine_method": "gating_transform",
            "layers": [100],
            "norm_layer": False
        },
        "SRL_BERT_gated_transform_100_norm": {
            "combine_method": "gating_transform",
            "layers": [100],
            "norm_layer": True
        },
    }

    for test in tests:
        print(f"\nTraining model {test}")
        model = SRL_BERT("bert-base-uncased", num_senses, num_roles, **tests[test])
        train(model, train_loader, val_loader, test_loader,
            epochs=10, init_lr=0.0001, scheduler_step=2, scheduler_gamma=0.9,
            device='cuda', name=test)
        # Save the model
        torch.save(model.state_dict(), "models/SRL_BERT_TEST.pt")
        print("-"*50)

if __name__ == '__main__':
    train_SRL()
