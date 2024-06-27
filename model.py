import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers import AutoTokenizer
import math


from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from dataloaders.UP_dataloader import roles

class GatedCombination(nn.Module):
    def __init__(self, hidden_size, transform=False):
        super(GatedCombination, self).__init__()
        self.gate = nn.Linear(2 * hidden_size, hidden_size)
        self.transform = transform
        if (transform): self.transform_layer = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, relation_hidden_state, word_hidden_states):
        combined = torch.cat([relation_hidden_state.expand_as(word_hidden_states), word_hidden_states], dim=-1)
        gating_scores = torch.sigmoid(self.gate(combined))

        if self.transform:
            transformed = torch.tanh(self.transform_layer(combined))
        else:
            transformed = relation_hidden_state.expand_as(word_hidden_states)
        
        return gating_scores * transformed + (1 - gating_scores) * word_hidden_states

class SRL_MODEL(nn.Module):
    def __init__(self, model_name, sense_classes, role_classes, 
                 combine_method='mean', norm_layer=False,
                 dim_reduction=0, relation_reduction=False,
                 role_layers=[], role_LSTM=False, train_encoder=True,
                 device='cuda'):
        '''
            Initialize the model

            Parameters:
                model_name (str): the name of the pretrained encoder model to use
                sense_classes (int): the number of classes for the senses classifier
                role_classes (int): the number of classes for the role classifier
                combine_method (str): the method to combine the hidden states of the words and relations, can be 'mean', 'concatenation', 'gating' or 'gating_transform'
                norm_layer (bool): whether to use a normalization layer after the combination
                dim_reduction (int): the size of the hidden states after the reduction
                relation_reduction (bool): whether to reduce the hidden states before the relational classifier
                role_layers (list): the sizes of the hidden layers of the role classifier
                role_LSTM (bool): whether to use an LSTM for the role classification (note: role_layers will be ignored, only its length is used)
                train_encoder (bool): whether to train the encoder model
                device (str): the device to use for the model
        
        '''
        super(SRL_MODEL, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        if not train_encoder:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.combine_method = combine_method  # 'mean', 'concatenation', 'gating'
        hidden_size = self.bert.config.hidden_size

        # senses_layers = [(self.bert.config.hidden_size, math.ceil(sense_classes/4)), 
        #                  (math.ceil(sense_classes/4), math.ceil(sense_classes/2)),
        #                  (math.floor(sense_classes/2), sense_classes)]
        # self.senses_classifier_layers = []
        # for i, (in_features, out_features) in enumerate(senses_layers):
        #     self.senses_classifier_layers.append(nn.Linear(in_features, out_features))
        #     if i < len(senses_layers) - 1:
        #         self.senses_classifier_layers.append(nn.ReLU())
        # self.senses_classifier = nn.Sequential(*self.senses_classifier_layers)

        role_size = hidden_size
        self.dim_reduction = (dim_reduction > 0)
        if self.dim_reduction:
            role_size = dim_reduction
            self.rel_reduction = nn.Sequential(nn.Linear(hidden_size, dim_reduction), nn.ReLU())
            self.word_reduction = nn.Sequential(nn.Linear(hidden_size, dim_reduction), nn.ReLU())

        self.rel_class_reduction = relation_reduction
        if self.rel_class_reduction:
            self.relational_classifier = nn.Linear(dim_reduction, 1)
        else:
            self.relational_classifier = nn.Linear(hidden_size, 1)

        if combine_method == 'gating_transform':
            self.combiner = GatedCombination(role_size, transform=True)
        elif combine_method == 'gating':
            self.combiner = GatedCombination(role_size, transform=False)

        # Configure input size based on combination method
        if combine_method == 'concatenation':
            role_classifer_input_size = 2 * role_size
        else: 
            role_classifer_input_size = role_size

        self.norm = norm_layer
        if norm_layer:
            # Instantiate a LayerNorm layer
            self.norm_layer = nn.LayerNorm(role_classifer_input_size)

        self.role_LSTM = role_LSTM
        if not role_LSTM:
            role_layers = [role_classifer_input_size] + role_layers + [role_classes]
            self.role_layers = []
            for i in range(len(role_layers) - 1):
                self.role_layers.append(nn.Linear(role_layers[i], role_layers[i+1]))
                if i < len(role_layers) - 2:
                    self.role_layers.append(nn.ReLU())
            self.role_classifier = nn.Sequential(*self.role_layers)
        else:
            if len(role_layers) > 0:
                print("Warning: role_layers values will be ignored when using an LSTM for the role classifier")
            # hoping to make them more stable
            self.role_classifier = nn.LSTM(role_classifer_input_size, role_classes, len(role_layers), batch_first=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.to(device)
        self.device = device

        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def rel_compute(self, hidden_states, word_ids):
        if self.rel_class_reduction and self.dim_reduction>0:
            hidden_states = self.rel_reduction(hidden_states)
            
        batch_size, seq_len, hidden_size = hidden_states.size()

        first_hidden_states = torch.zeros((batch_size, max(word_ids)+1, hidden_size)).to(hidden_states.device)
        seen_words = set()
        for j, word_id in enumerate(word_ids):
            if word_id != -1 and word_id not in seen_words:
                first_hidden_states[0, word_id] = hidden_states[0, j]
                seen_words.add(word_id)

        relational_logits = self.relational_classifier(first_hidden_states).squeeze(-1)
        return relational_logits

    def sense_compute(self, hidden_states, relations_positions):
        phrase_indices = [[i]*len(pos) for i, pos in enumerate(relations_positions)]
        phrase_indices = [i for sublist in phrase_indices for i in sublist]
        relations_indices = [pos for positions in relations_positions for pos in positions]
        
        senses_logits = self.senses_classifier(hidden_states[phrase_indices, relations_indices])
        return senses_logits
    
    def role_compute(self, hidden_states, relations_positions, word_ids):
        batch_size, seq_len, hidden_size = hidden_states.size()
        # Extract the first hidden state for each word
        first_hidden_states = torch.zeros((batch_size, max([max(words_id) for words_id in word_ids])+1, hidden_size)).to(hidden_states.device)
        
        prev_word_id = -1
        for i in range(batch_size):
            for j, word_id in enumerate(word_ids[i]):
                if word_id != -1 and word_id != prev_word_id:
                    # breakpoint()
                    first_hidden_states[i, word_id] = hidden_states[i, j]
                    prev_word_id = word_id

        # Combine the hidden states
        results = []
        # breakpoint()
        for i in range(batch_size):
            relation_hidden_states = []
            for pos in relations_positions[i]:
                if pos is not None and pos < seq_len:
                    relation_hidden_state = hidden_states[i, pos]
                    relation_hidden_state = self.rel_reduction(relation_hidden_state) if self.dim_reduction>0 else relation_hidden_state
                    
                    word_hidden_states = first_hidden_states[i, [word_id for word_id in set(word_ids[i]) if word_id != -1]]
                    word_hidden_states = self.word_reduction(word_hidden_states) if self.dim_reduction>0 else word_hidden_states

                    if(self.combine_method == 'mean'):
                        # mean between the two states
                        combined_states = (word_hidden_states + relation_hidden_state) / 2
                    elif(self.combine_method == 'concatenation'):
                        combined_states = torch.cat([relation_hidden_state.expand_as(word_hidden_states), word_hidden_states], dim=-1)
                    elif(self.combine_method == 'gating' or self.combine_method == 'gating_transform'):
                        combined_states = self.combiner(relation_hidden_state, word_hidden_states)
                    
                    # breakpoint()
                    
                    relation_hidden_states.append(combined_states)

            if relation_hidden_states:
                relation_hidden_states = torch.stack(relation_hidden_states)
                relation_hidden_states = self.norm_layer(relation_hidden_states) if self.norm else relation_hidden_states

                if not self.role_LSTM:
                    role_logits = self.role_classifier(relation_hidden_states)
                else:
                    role_logits, _ = self.role_classifier(relation_hidden_states)
                
                results.append(role_logits)
            else:
                print("No relations found in the sentence")

        return results

    def forward(self, input_ids, attention_mask, relations_positions, word_ids):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # Compute the logits for the relational classifier
        relational_class_input = self.rel_reduction(hidden_states) if self.dim_reduction>0 else hidden_states
        relational_logits = self.relational_classifier(relational_class_input).squeeze(-1)
        
        # Compute the logits for the senses classifier
        # senses_logits = self.sense_compute(hidden_states, relations_positions)
        senses_logits = None
        
        # Compute the logits for the role classifier
        results = self.role_compute(hidden_states, relations_positions, word_ids)
        
        return relational_logits, senses_logits, results
    
    def inference(self,text):
        self.eval()

        # tokenize the text as in the training (TreebankWordTokenizer)
        tokenizer = TreebankWordTokenizer()
        text = tokenizer.tokenize(text)
        text = " ".join(text)

        with torch.no_grad():
            tokenized_text = self.tokenizer(text, return_tensors='pt')
            input_ids = tokenized_text['input_ids']
            attention_mask = tokenized_text['attention_mask']
            word_ids = tokenized_text.word_ids()
            word_ids = [(word_id if word_id is not None else -1) for word_id in word_ids]
            # brutta soluzione a J.
            delta = 0
            seen_ids = set()
            for i, word_id in enumerate(word_ids):
                if word_id <= 0: continue
                start, end = tokenized_text.word_to_chars(word_id)
                if text[start-1] != ' ' and word_id not in seen_ids:
                    delta += 1
                    seen_ids.add(word_id)
                word_ids[i] = word_id - delta

            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state

            relational_logits = self.rel_compute(hidden_states, word_ids).squeeze(0)
            #apply sigmoid
            relational_probabilities = torch.sigmoid(relational_logits)
            relation_positions = [i for i in range(len(relational_probabilities)) if relational_probabilities[i] > 0.75]
            relation_positions = [word_ids.index(i) for i in relation_positions]

            # senses_logits = self.sense_compute(hidden_states, [relation_positions])
            senses_logits = None

            results = self.role_compute(hidden_states, [relation_positions], [word_ids])
            # results = [result.squeeze(0) for result in results]

        return  relational_logits, senses_logits, results

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

    relation_positions = [i for i in range(len(relational_logits)) if nn.Sigmoid()(relational_logits[i]) > 0.9]

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
    from utils import get_dataloaders
    # _, _, _, num_senses, num_roles = get_dataloaders("datasets/preprocessed/", batch_size=32, shuffle=True)


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