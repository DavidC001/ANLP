import sys
sys.path.append('.')
from model import SRL_MODEL
from train.utils import get_dataloaders
from dataloaders.NomBank_dataloader import roles as NOM_roles
from dataloaders.UP_dataloader import roles as UP_roles
from train.functions import eval_step
import torch
import json
import os
from tqdm import tqdm
import argparse

torch.manual_seed(0)

def train_SRL(top=True):
    #cycle over all models in the models folder
    
    results = {}
    for model_name in tqdm(os.listdir('models')):
        if model_name.endswith('.pt'):
            print(f"\nprocessing model {model_name}")
            model_config = "models/"+model_name[:-3]+'.json'
            with open(model_config, "r") as f:
                config = json.load(f)

            model = SRL_MODEL(**config)
            model.load_state_dict(torch.load(f"models/{model_name}"))
            num_params = sum(p[1].numel() for p in model.named_parameters())
            num_params_classifiers = sum(p[1].numel() for p in model.named_parameters() if 'bert' not in p[0])

            if(config["role_classes"] == len(UP_roles)-2):
                dataset = "UP"
            elif(config["role_classes"] == len(NOM_roles)-2):
                dataset = "NOM"

            _,_,test,_,_ = get_dataloaders("datasets/preprocessed/", batch_size=32, shuffle=False, model_name=config["model_name"], dataset=dataset)

            result = eval_step(model, test, l2_lambda=0, F1_loss_power=0, top=top)
            results[model_name] = {"result":result, "params": num_params, "params_class":num_params_classifiers}
    
    # save results to a file
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)



if __name__ == '__main__':
    #get from arguments top or not
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=bool, default=True)
    args = parser.parse_args()
    train_SRL(args.top)
