import sys

sys.path.append(".")
from model import SRL_MODEL
from train.utils import get_dataloaders
from train.functions import train
import torch
import os
import json

torch.manual_seed(0)


def train_SRL():
    tests = {
        "small_distil": {
            "model_name": "distilbert-base-uncased",  # name of the encoder model to use
            
            "combine_method": "multiHeadAttention",  # how to combine the predicate and word representations
            "attention_heads": 1,  # number of attention heads, if multiHeadAttention is used
            "FFN_layers": 2,  # number of layers in the FFN for the combined representation
            "norm_layer": True,  # whether to apply layer normalization
            
            "role_RNN": True,  # whether to use a LSTM layer in the role classifier
            "RNN_type": "GRU",  # type of RNN layer

            "proj_dim": 256,  # dimension of the projection layer
            "relation_proj": True,  # whether to project the relation representation
            
            "train_encoder": True,  # whether to train the encoder
            "train_embedding_layer": True,  # whether to train the embedding layer
            "dropout_prob": 0.5,  # dropout rate
            "variational_dropout": True,  # whether to use variational dropout
        },
    }

    training_params = {
        "epochs": 100,  # number of epochs
        "init_lr": 1e-5,  # initial learning rate
        "lr_encoder": 1e-7,  # learning rate for the encoder
        "F1_loss_power": 0.5,  # power to use in the F1 loss
        "patience": 3,  # patience for the early stopping
        "role_threshold": 0.5,  # threshold for the role classifier
        "group_roles": True,  # whether to group the roles
        "top": False,  # whether to use the top-k roles
        "weight_role_labels": False,  # whether to weight the role labels
        "noise": 0.1,  # noise to add to the input
        "random_sostitution_prob": 0.2,  # probability of randomly substituting a word with UNK token
        "device": "cuda" if torch.cuda.is_available() else "cpu",  # device to use
    }

    # Choose dataset UP or NOM
    dataset = "UP"

    for test in tests:
        batch_size = (
            16
            if ("large" in tests[test]["model_name"] and tests[test]["train_encoder"])
            else 32
        )
        train_loader, val_loader, test_loader, num_senses, num_roles = get_dataloaders(
            "datasets/preprocessed/",
            batch_size=batch_size,
            shuffle=True,
            model_name=tests[test]["model_name"],
            dataset=dataset,
        )
        print(f"\nTraining model {test}")
        tests[test]["sense_classes"] = num_senses
        tests[test]["role_classes"] = num_roles

        model = SRL_MODEL(**tests[test], device=training_params["device"])
        print(
            f"Total number of parameters in the model: {sum(p[1].numel() for p in model.named_parameters())}"
        )
        print(
            f"Number of parameters in the classifiers: {sum(p[1].numel() for p in model.named_parameters() if 'bert' not in p[0])}"
        )

        # load the model from a previous training
        if os.path.exists(f"models/{test}.pt"):
            model.load_state_dict(torch.load(f"models/{test}.pt"))
            print("Model loaded from previous training")


        # Save json with the test parameters of the model and the training parameters
        with open(f"models/{test}.json", "w") as f:
            json.dump(tests[test], f)
        with open(f"models/{test}_training_setup.json", "w") as f:
            json.dump(training_params, f)

        train(
            model,
            train_loader,
            val_loader,
            test_loader,
            name=test,
            dataset=dataset,
            **training_params,
        )

        print("-" * 50)


if __name__ == "__main__":
    train_SRL()
