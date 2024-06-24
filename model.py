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

class SRL_BERT(nn.Module):
    def __init__(self, model_name, sense_classes, role_classes, role_layers, device='cuda', combine_method='mean', norm_layer=False):
        '''
            Initialize the model

            Parameters:
                model_name (str): the name of the pretrained encoder model to use
                sense_classes (int): the number of classes for the senses classifier
                role_classes (int): the number of classes for the role classifier
                role_layers (list): a list of integers representing the number of neurons in each layer of the role classifier
                device (str): the device to use for training
                combine_method (str): the method to combine the hidden states of the relation and the word
                norm_layer (bool): whether to use a BatchNorm layer for role classification
        
        '''
        super(SRL_BERT, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.combine_method = combine_method  # 'mean', 'concatenation', 'gating'
        hidden_size = self.bert.config.hidden_size

        self.relational_classifier = nn.Linear(hidden_size, 1)

        # senses_layers = [(self.bert.config.hidden_size, math.ceil(sense_classes/4)), 
        #                  (math.ceil(sense_classes/4), math.ceil(sense_classes/2)),
        #                  (math.floor(sense_classes/2), sense_classes)]
        # self.senses_classifier_layers = []
        # for i, (in_features, out_features) in enumerate(senses_layers):
        #     self.senses_classifier_layers.append(nn.Linear(in_features, out_features))
        #     if i < len(senses_layers) - 1:
        #         self.senses_classifier_layers.append(nn.ReLU())
        # self.senses_classifier = nn.Sequential(*self.senses_classifier_layers)

        if combine_method == 'gating_transform':
            self.combiner = GatedCombination(hidden_size, transform=True)
        elif combine_method == 'gating':
            self.combiner = GatedCombination(hidden_size, transform=False)


        # Configure input size based on combination method
        if combine_method == 'concatenation':
            role_classifer_input_size = 2 * hidden_size
        else:  # concatenation
            role_classifer_input_size = hidden_size

        self.norm = norm_layer
        if norm_layer:
            # Instantiate a LayerNorm layer
            self.norm_layer = nn.LayerNorm(role_classifer_input_size)

        role_layers = [role_classifer_input_size] + role_layers + [role_classes]
        self.role_layers = []
        for i in range(len(role_layers) - 1):
            self.role_layers.append(nn.Linear(role_layers[i], role_layers[i+1]))
            if i < len(role_layers) - 2:
                self.role_layers.append(nn.ReLU())
        self.role_classifier = nn.Sequential(*self.role_layers)

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.to(device)
        self.device = device

        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def rel_compute(self, hidden_states, word_ids):
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
                    
                    word_hidden_states = first_hidden_states[i, [word_id for word_id in set(word_ids[i]) if word_id != -1]]
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

                role_logits = self.role_classifier(relation_hidden_states)
                # breakpoint()
                results.append(role_logits)
            else:
                raise ValueError("No relations found in the sentence")

        return results

    def forward(self, input_ids, attention_mask, relations_positions, word_ids):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # Compute the logits for the relational classifier
        relational_logits = self.relational_classifier(hidden_states).squeeze(-1)
        
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
        print(text)

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

        return relational_logits, senses_logits, results

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

    print("Role logits:")
    for i, phrase_role_logits in enumerate(results):
        for j, relation_role_logits in enumerate(phrase_role_logits):
            print(f"\tRelation {j}")
            for k, role_logits in enumerate(relation_role_logits):
                # breakpoint()
                predicted_roles = [
                    f"{roles[q+1]} {nn.Sigmoid()(role_logits[q]):.2f}"
                    for q in range(len(role_logits))
                    if nn.Sigmoid()(role_logits[q]) > 0.6 and q != 0
                ]

                print(f"\t\tWord: {text[k]} - predicted roles: {predicted_roles}")
                
                # for q, role_logit in enumerate(role_logits):
                #     print(f"\t\t\tRole: {roles[q+1]} - Probability: {nn.Sigmoid()(role_logit)}")

if __name__ == '__main__':
    from utils import get_dataloaders
    _, _, _, num_senses, num_roles = get_dataloaders("datasets/preprocessed/", batch_size=32, shuffle=True)

    model = SRL_BERT("bert-base-uncased", num_senses, num_roles, [50], device='cuda', combine_method='gating_transform', norm_layer=True)
    model.load_state_dict(torch.load("models/SRL_BERT_TEST_gate_norm.pt"))
    text = "Fausto eats polenta at the beach while sipping beer."

    while text != "exit":
        relational_logits, senses_logits, results = model.inference(text)
        print_results(relational_logits, senses_logits, results, text)
        text = input("Insert a sentence to analyze (type 'exit' to quit): ")