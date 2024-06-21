import torch
from torch import nn
from transformers import AutoModel
import math


import torch
import torch.nn as nn
from transformers import AutoModel

class SRL_BERT(nn.Module):
    def __init__(self, model_name, sense_classes, role_classes, device):
        super(SRL_BERT, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.relational_classifier = nn.Linear(self.bert.config.hidden_size, 1)
        senses_layers = [(self.bert.config.hidden_size, math.ceil(sense_classes/4)), 
                         (math.ceil(sense_classes/4), math.ceil(sense_classes/2)),
                         (math.floor(sense_classes/2), sense_classes)]
        self.senses_classifier_layers = []
        for i, (in_features, out_features) in enumerate(senses_layers):
            self.senses_classifier_layers.append(nn.Linear(in_features, out_features))
            if i < len(senses_layers) - 1:
                self.senses_classifier_layers.append(nn.ReLU())
        self.senses_classifier = nn.Sequential(*self.senses_classifier_layers)
        self.role_classifier = nn.Linear(self.bert.config.hidden_size * 2, role_classes)

        self.to(device)
        self.device = device

    def forward(self, input_ids, attention_mask, relations_positions, word_ids):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        relational_logits = self.relational_classifier(hidden_states).squeeze(-1)
        i = 0
        phrase_indices = [[i]*len(pos) for i, pos in enumerate(relations_positions)]
        phrase_indices = [i for sublist in phrase_indices for i in sublist]
        relations_indices = [pos for positions in relations_positions for pos in positions]
        # breakpoint()
        senses_logits = self.senses_classifier(hidden_states[phrase_indices, relations_indices])
        
        # Extract the first hidden state for each word
        batch_size, seq_len, hidden_size = hidden_states.size()
        first_hidden_states = torch.zeros((batch_size, max([max(words_id) for words_id in word_ids])+1, hidden_size)).to(hidden_states.device)
        
        for i in range(batch_size):
            for j, word_id in enumerate(word_ids[i]):
                if word_id != -1:
                    first_hidden_states[i, word_id] = hidden_states[i, j]

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
        
        return relational_logits, senses_logits, results
