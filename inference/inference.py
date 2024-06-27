import torch
from nltk.tokenize import TreebankWordTokenizer
import torch.nn as nn
import sys
sys.path.append('.')

from model import SRL_MODEL
from dataloaders.UP_dataloader import roles


def print_results(relational_logits, senses_logits, results, text):
    # tokenize the text as in the training (TreebankWordTokenizer)
    tokenizer = TreebankWordTokenizer()
    text = tokenizer.tokenize(text)
    text = " ".join(text)

    print(f"Text: {text}")

    text = text.split()

    print("Relational logits:")
    for i, prob in enumerate(relational_logits):
        print(f"\tWord: {text[i]} - Probability: {nn.Sigmoid()(prob)}")

    # print("Senses logits:")
    # print(senses_logits)

    relation_positions = [i for i in range(len(relational_logits)) if nn.Sigmoid()(relational_logits[i]) > 0.75]

    print("Role logits:")   
    for i, phrase_role_logits in enumerate(results):
        for j, relation_role_logits in enumerate(phrase_role_logits):
            print(f"\tRelation {j} - Word: {text[relation_positions[j]]}")
            for k, role_logits in enumerate(relation_role_logits):
                # breakpoint()
                predicted_roles = [
                    f"{roles[q+2]} {nn.Sigmoid()(role_logits[q]):.2f}"
                    for q in range(len(role_logits))
                    if nn.Sigmoid()(role_logits[q]) > 0.5 and q != 0
                ]

                print(f"\t\tWord: {text[k]} - predicted roles: {predicted_roles}")
                
                # for q, role_logit in enumerate(role_logits):
                #     print(f"\t\t\tRole: {roles[q+1]} - Probability: {nn.Sigmoid()(role_logit)}")

if __name__ == '__main__':
    name = input("Insert the name of the model to load: ")
    #read configuration from file json
    import json
    with open(f"models/{name}.json", "r") as f:
        config = json.load(f)
    
    model = SRL_MODEL(**config)
    model.load_state_dict(torch.load(f"models/{name}.pt"))
    text = "Fausto eats polenta at the beach while sipping beer."

    #print number of parameters in model excluding bert
    print(f"Number of parameters in the model: {sum(p[1].numel() for p in model.named_parameters() if 'bert' not in p[0])}")

    while text != "exit":
        relational_logits, senses_logits, results = model.inference(text)
        print_results(relational_logits, senses_logits, results, text)
        text = input("Insert a sentence to analyze (type 'exit' to quit): ")