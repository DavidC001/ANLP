import sys
sys.path.append('.')
from model import SRL_MODEL
from train.utils import get_dataloaders
from train.functions import train
import torch
import json

torch.manual_seed(0)

def train_SRL():
    tests = {
        "TEST": {
            "model_name": "distilbert/distilbert-base-uncased", # name of the encoder model to use
            "combine_method": "gating", # how to combine the predicate and word representations
            "role_layers": [100], # hidden dimensions of the role classifier
            "norm_layer": True, # whether to apply layer normalization
            "proj_dim": 200, # dimension of the projection layer
            "relation_proj": True, # whether to project the relation representation
            "train_encoder": True, # whether to train the encoder
            "dropout_prob": 0.2, # dropout rate
        },
    }

    # Choose dataset UP or NOM
    dataset = "UP"

    for test in tests:
        batch_size = 32 if ("large" in tests[test]["model_name"] and tests[test]["train_encoder"]) else 64
        train_loader, val_loader, test_loader, num_senses, num_roles = get_dataloaders(
            "datasets/preprocessed/", 
            batch_size=batch_size, shuffle=True, 
            model_name=tests[test]["model_name"],
            dataset=dataset
        )
        print(f"\nTraining model {test}")
        tests[test]["sense_classes"] = num_senses
        tests[test]["role_classes"] = num_roles

        model = SRL_MODEL(**tests[test])
        print(f"Total number of parameters in the model: {sum(p[1].numel() for p in model.named_parameters())}")
        print(f"Number of parameters in the classifiers: {sum(p[1].numel() for p in model.named_parameters() if 'bert' not in p[0])}")

        train(model, train_loader, val_loader, test_loader,
            epochs=10, init_lr=0.001, lr_encoder=1e-5, l2_lambda=1e-5, F1_loss_power=0,
            role_threshold=0.5, group_roles=True, top=False, 
            noise=0.1, noise_prob=0.2,
            device='cuda', name=test, dataset=dataset)
        
        # Save the model
        torch.save(model.state_dict(), f"models/{test}.pt")
        # Save json with the test parameters
        with open(f"models/{test}.json", "w") as f:
            json.dump(tests[test], f, indent=4)

        print("-"*50)

if __name__ == '__main__':
    train_SRL()
