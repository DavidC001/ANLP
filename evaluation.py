import sys
sys.path.append('.')
from model import SRL_MODEL
from train.utils import get_dataloaders
from torch.utils.data import DataLoader
from dataloaders.NomBank_dataloader import roles as NOM_roles
from dataloaders.UP_dataloader import roles as UP_roles
from train.functions import eval_step, top_span
from sklearn.metrics import precision_recall_fscore_support
import torch
import json
import os
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
    

torch.manual_seed(0)

def evaluation_metrics_entire_dataset(model:SRL_MODEL, loader:DataLoader, top:bool=False, threshold:float=0.5):
    all_preds = []
    all_labels = []

    model.eval()
    device = model.device

    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(device)
            word_ids = batch['word_ids']
            attention_masks = batch['attention_masks'].to(device)
            relations = batch['relation_position']
            role_labels = batch['role_labels']
            # rel_senses = batch['senses'].to(device)

            relation_label_masks = batch['relation_label_mask'].to(device)
            relation_labels = batch['relation_label'].to(device)

            senses_labels = batch['senses_labels'].to(device)

            relational_logits, senses_logits, role_results = model(input_ids, attention_masks, relations, word_ids)

            for i in range(len(role_results)):
                role_logits = role_results[i]
                role_label = role_labels[i]

                if (top): role_preds = top_span(role_logits, threshold=threshold)
                else: role_preds = (torch.sigmoid(role_logits) > threshold).float()

                for j in range(role_label.shape[0]): # for each relation
                    for k in range(role_label.shape[1]): # for each word
                        all_labels.append(role_label[j][k].cpu().numpy())
                        all_preds.append(role_preds[j][k].cpu().numpy())

        
        micro_role_precision, micro_role_recall, micro_role_f1, _ = precision_recall_fscore_support(all_labels, all_preds, zero_division=1, average='micro')
        macro_role_precision, macro_role_recall, macro_role_f1, _ = precision_recall_fscore_support(all_labels, all_preds, zero_division=1, average='macro')
        return {
            "micro_precision": micro_role_precision,
            "micro_recall": micro_role_recall,
            "micro_f1": micro_role_f1,
            "macro_precision": macro_role_precision,
            "macro_recall": macro_role_recall,
            "macro_f1": macro_role_f1
        }



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

            result_dataset = evaluation_metrics_entire_dataset(model,test,top=top, threshold=threshold)
            result_batch = eval_step(model, test, l2_lambda=0, F1_loss_power=0, top=top, role_threshold=threshold, device=device)

            #convert the tensors to floats to be able to save the results to a json file
            for key, value in result_batch.items():
                if isinstance(value, torch.Tensor):
                    result_batch[key] = value.tolist()
            for key, value in result_dataset.items():
                if isinstance(value, torch.Tensor):
                    result_dataset[key] = value.tolist()

            results[model_name] = {"result":result_batch, "role_results_dataset":result_dataset, "params": num_params, "params_class":num_params_classifiers}
    
            # save results to a file
            with open(f"results_{"top" if top else "concat"}_{threshold}.json", "w") as f:
                # save the results to a json file
                json.dump(results, f)

    #display graph with role F1 for y and number of parameters for x for UP
    # plt.figure()
    # for model_name, data in results.items():
    #     if "NOM" not in model_name:
    #         plt.scatter(data["params"], data["role_results_dataset"]["micro_f1"], label=model_name)

    # plt.xlabel("Number of parameters UP dataset")
    # plt.ylabel("Role F1")
    # plt.legend()
    # plt.show()

    #display graph with role F1 for y and number of parameters in classifiers for x for UP
    plt.figure()
    for model_name, data in results.items():
        if "NOM" not in model_name:
            plt.scatter(data["params_class"], data["role_results_dataset"]["micro_f1"], label=model_name)

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
                f.write(f"{model_name} & {data['role_results_dataset']['micro_f1']:.3f} & {data['result']['role_loss']:.3f} & {data['result']['rel_f1']:.3f} & {data['result']['rel_loss']:.3f} \\\\ \n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


    # for NOM
    # plt.figure()
    # for model_name, data in results.items():
    #     if "NOM" in model_name:
    #         plt.scatter(data["params"], data["role_results_dataset"]["micro_f1"], label=model_name)

    # plt.xlabel("Number of parameters NOM dataset")
    # plt.ylabel("Role F1")
    # plt.legend()
    # plt.show()

    plt.figure()
    for model_name, data in results.items():
        if "NOM" in model_name:
            plt.scatter(data["params_class"], data["role_results_dataset"]["micro_f1"], label=model_name)

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
                f.write(f"{model_name} & {data['micro_f1']['micro_f1']:.3f} & {data['result']['role_loss']:.3f} & {data['result']['rel_f1']:.3f} & {data['result']['rel_loss']:.3f} \\\\ \n")
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
