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

    # load the already evaluated models from results.json
    if os.path.exists(f"results_{top}.json"):
        with open(f"results_{top}.json", "r") as f:
            results = json.load(f)
    
    for model_name in tqdm(os.listdir('models')):
        if model_name in results.keys():
            print(f"\nmodel {model_name} already evaluated, skipping")
        if model_name.endswith('.pt') and model_name not in results.keys():
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

            #convert the tensors to floats to be able to save the results to a json file
            for key, value in result.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value.tolist()

            results[model_name] = {"result":result, "params": num_params, "params_class":num_params_classifiers}
    
            # save results to a file
            with open(f"results_{top}.json", "w") as f:
                # save the results to a json file
                json.dump(results, f)

    #display graph with role F1 for y and number of parameters for x
    import matplotlib.pyplot as plt
    plt.figure()
    for model_name, data in results.items():
        plt.scatter(data["params"], data["result"]["role_f1"], label=model_name)

    plt.xlabel("Number of parameters")
    plt.ylabel("Role F1")
    plt.legend()
    plt.show()

    plt.figure()
    for model_name, data in results.items():
        plt.scatter(data["params_class"], data["result"]["role_f1"], label=model_name)

    plt.xlabel("Number of parameters in classifiers")
    plt.ylabel("Role F1")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    #get from arguments top or not
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=bool, default=True)
    args = parser.parse_args()
    train_SRL(args.top)
