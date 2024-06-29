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
        "SRL_NOM_XML_ROBERTA_CONLL_gated_redboth100_100_norm_cosineLR_weightedLoss": {
            "model_name": "FacebookAI/xlm-roberta-large-finetuned-conll03-english",
            "combine_method": "gating",
            "role_layers": [100],
            "norm_layer": True,
            "dim_reduction": 100,
            "relation_reduction": True,
            "train_encoder": True
        },
    }

    dataset = "NOM"

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
            epochs=10, init_lr=0.001, lr_encoder=1e-5, l2_lambda=1e-5, F1_loss_power=1,
            device='cuda', name=test, dataset=dataset)
        
        # Save the model
        torch.save(model.state_dict(), f"models/{test}.pt")
        # Save json with the test parameters
        with open(f"models/{test}.json", "w") as f:
            json.dump(tests[test], f, indent=4)

        print("-"*50)

if __name__ == '__main__':
    train_SRL()
