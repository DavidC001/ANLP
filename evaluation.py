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
import matplotlib.pyplot as plt
    

torch.manual_seed(0)

def train_SRL(top=True, threshold=0.5):
    #cycle over all models in the models folder
    results = {}

    # load the already evaluated models from results.json
    if os.path.exists(f"results_{"top" if top else "concat"}_{threshold}.json"):
        with open(f"results_{"top" if top else "concat"}_{threshold}.json", "r") as f:
            results = json.load(f)
    
    for model_name in tqdm(os.listdir('models')):
        if model_name in results.keys():
            print(f"\nmodel {model_name} already evaluated, skipping")
        if model_name.endswith('.pt') and model_name not in results.keys():
            print(f"\nprocessing model {model_name}")
            model_config = "models/"+model_name[:-3]+'.json'
            with open(model_config, "r") as f:
                config = json.load(f)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            config["device"] = device

            model = SRL_MODEL(**config)
            model.load_state_dict(torch.load(f"models/{model_name}"))
            num_params = sum(p[1].numel() for p in model.named_parameters())
            num_params_classifiers = sum(p[1].numel() for p in model.named_parameters() if 'bert' not in p[0])

            if(config["role_classes"] == len(UP_roles)-2):
                dataset = "UP"
            elif(config["role_classes"] == len(NOM_roles)-2):
                dataset = "NOM"

            _,_,test,_,_ = get_dataloaders("datasets/preprocessed/", batch_size=32, shuffle=False, model_name=config["model_name"], dataset=dataset)

            result = eval_step(model, test, l2_lambda=0, F1_loss_power=0, top=top, role_threshold=threshold, device=device)

            #convert the tensors to floats to be able to save the results to a json file
            for key, value in result.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value.tolist()

            results[model_name] = {"result":result, "params": num_params, "params_class":num_params_classifiers}
    
            # save results to a file
            with open(f"results_{"top" if top else "concat"}_{threshold}.json", "w") as f:
                # save the results to a json file
                json.dump(results, f)

    #display graph with role F1 for y and number of parameters for x for UP
    # plt.figure()
    # for model_name, data in results.items():
    #     if "NOM" not in model_name:
    #         plt.scatter(data["params"], data["result"]["role_f1"], label=model_name)

    # plt.xlabel("Number of parameters UP dataset")
    # plt.ylabel("Role F1")
    # plt.legend()
    # plt.show()

    #display graph with role F1 for y and number of parameters in classifiers for x for UP
    plt.figure()
    for model_name, data in results.items():
        if "NOM" not in model_name:
            plt.scatter(data["params_class"], data["result"]["role_f1"], label=model_name)

    plt.xlabel("Number of parameters in classifiers UP dataset")
    plt.ylabel("Role F1")
    plt.legend()
    plt.show()

    # make it into a latex table for the report, one having the role F1, precision, recall and accuracy and the other with F1, precision, recall and accuracy for the relations
    with open(f"results_table_UP_{top}.tex", "w") as f:
        f.write("\\begin{table}\n")
        f.write("\\begin{tabular}{l|c|c|c|c}\n")
        f.write("\\hline\nModel & \\multicolumn{2}{c|}{Role} & \\multicolumn{2}{c}{Pred} \\\\ \n\\hline\n& F1 & Loss & F1 & Loss \\\\\n\\hline\n")
        for model_name, data in results.items():
            if "NOM" not in model_name:
                f.write(f"{model_name} & {data['result']['role_f1']:.3f} & {data['result']['role_loss']:.3f} & {data['result']['rel_f1']:.3f} & {data['result']['rel_loss']:.3f} \\\\ \n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


    # for NOM
    # plt.figure()
    # for model_name, data in results.items():
    #     if "NOM" in model_name:
    #         plt.scatter(data["params"], data["result"]["role_f1"], label=model_name)

    # plt.xlabel("Number of parameters NOM dataset")
    # plt.ylabel("Role F1")
    # plt.legend()
    # plt.show()

    plt.figure()
    for model_name, data in results.items():
        if "NOM" in model_name:
            plt.scatter(data["params_class"], data["result"]["role_f1"], label=model_name)

    plt.xlabel("Number of parameters in classifiers NOM dataset")
    plt.ylabel("Role F1")
    plt.legend()
    plt.show()

    print("\n\n")
    # make it into a latex table for the report, one having the role F1, precision, recall and accuracy and the other with F1, precision, recall and accuracy for the relations
    with open(f"results_table_NOM_{top}.tex", "w") as f:
        f.write("\\begin{table}\n")
        f.write("\\begin{tabular}{l|c|c|c|c}\n")
        f.write("\\hline\nModel & \\multicolumn{2}{c|}{Role} & \\multicolumn{2}{c}{Pred} \\\\ \n\\hline\n& F1 & Loss & F1 & Loss \\\\\n\\hline\n")
        for model_name, data in results.items():
            if "NOM" in model_name:
                f.write(f"{model_name} & {data['result']['role_f1']:.3f} & {data['result']['role_loss']:.3f} & {data['result']['rel_f1']:.3f} & {data['result']['rel_loss']:.3f} \\\\ \n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")



if __name__ == '__main__':
    #get from arguments top or not
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", action="store_true")
    parser.add_argument("--concat", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.concat:
        print(f"\n\nRunning evaluation with concat strategy and threshold {args.threshold}")
        train_SRL(top=False, threshold=args.threshold)
    if args.top:
        print(f"\n\nRunning evaluation with top strategy and threshold {args.threshold}")
        train_SRL(top=True, threshold=args.threshold)

    if not args.concat and not args.top:
        print("Please specify either --top or --concat")
