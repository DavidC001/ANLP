import torch
from torch import nn
from transformers import AutoModel
from transformers import AutoTokenizer
import math

from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from dataloaders.UP_dataloader import roles

class SRL_BERT(nn.Module):
    def __init__(self, model_name, sense_classes,role_classes, role_layers, device):
        super(SRL_BERT, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        self.relational_classifier = nn.Linear(self.bert.config.hidden_size, 1)

        # senses_layers = [(self.bert.config.hidden_size, math.ceil(sense_classes/4)), 
        #                  (math.ceil(sense_classes/4), math.ceil(sense_classes/2)),
        #                  (math.floor(sense_classes/2), sense_classes)]
        # self.senses_classifier_layers = []
        # for i, (in_features, out_features) in enumerate(senses_layers):
        #     self.senses_classifier_layers.append(nn.Linear(in_features, out_features))
        #     if i < len(senses_layers) - 1:
        #         self.senses_classifier_layers.append(nn.ReLU())
        # self.senses_classifier = nn.Sequential(*self.senses_classifier_layers)

        role_layers = [self.bert.config.hidden_size * 2] + role_layers + [role_classes]
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
                    
                    combined_states = torch.cat(
                        [first_hidden_states[i, word_id].unsqueeze(0) for word_id in set(word_ids[i]) if word_id != -1], 
                        dim=0
                    )

                    # breakpoint()

                    combined_states = torch.cat(
                        (combined_states, relation_hidden_state.unsqueeze(0).expand_as(combined_states)), 
                        dim=-1
                    )
                    relation_hidden_states.append(combined_states)
            
            if relation_hidden_states:
                relation_hidden_states = torch.stack(relation_hidden_states)
                # breakpoint()
                role_logits = self.role_classifier(relation_hidden_states).view(len(relations_positions[i]), -1, self.role_classifier.out_features)
                # breakpoint()
                results.append(role_logits)
            else:
                results.append(torch.empty(0, self.role_classifier.out_features).to(hidden_states.device))

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

        with torch.no_grad():
            tokenized_text = self.tokenizer(text, return_tensors='pt')
            input_ids = tokenized_text['input_ids']
            attention_mask = tokenized_text['attention_mask']
            word_ids = tokenized_text.word_ids()
            word_ids = [(word_id if word_id is not None else -1) for word_id in word_ids]

            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state

            relational_logits = self.rel_compute(hidden_states, word_ids).squeeze(0)
            #apply sigmoid
            relational_probabilities = torch.sigmoid(relational_logits)
            relation_positions = [i for i in range(len(relational_probabilities)) if relational_probabilities[i] > 0.5]

            # senses_logits = self.sense_compute(hidden_states, [relation_positions])
            senses_logits = None

            results = self.role_compute(hidden_states, [relation_positions], [word_ids])
            results = [result.squeeze(0) for result in results]

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
                print(f"\t\tWord: {text[k]} - predicted roles: {[roles[q+1] for q in range(len(role_logits)) if nn.Sigmoid()(role_logits[q]) > 0.5]}")
                # for q, role_logit in enumerate(role_logits):
                #     print(f"\t\t\tRole: {roles[q+1]} - Probability: {nn.Sigmoid()(role_logit)}")

if __name__ == '__main__':
    from utils import get_dataloaders
    _, _, _, num_senses, num_roles = get_dataloaders("datasets/preprocessed/", batch_size=32, shuffle=True)

    model = SRL_BERT("bert-base-uncased", num_senses, [num_roles], device='cuda')
    model.load_state_dict(torch.load("models/SRL_BERT_TEST_bella.pt"))
    text = "Fausto eats polenta."
    relational_logits, senses_logits, results = model.inference(text)
    print_results(relational_logits, senses_logits, results, text)