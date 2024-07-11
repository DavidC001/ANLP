import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers import AutoTokenizer
import math


from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer

class GatedCombination(nn.Module):
    """
        Combines the hidden states of the words and the relations using a gating mechanism
    """
    def __init__(self, hidden_size, transform=False):
        """
            Initialize the module

            Parameters:
                hidden_size (int): the size of the hidden states
                transform (bool): false combines directly, true applies a transformation before combining with the second state
        """
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

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        
        # Forward propagate LSTM
        out, _ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden states for each word
        out = self.fc(out)
        return out

class SRL_MODEL(nn.Module):
    def __init__(self, model_name, sense_classes, role_classes, 
                 combine_method='mean', norm_layer=False,
                 proj_dim=0, relation_proj=False,
                 role_layers=[], role_LSTM=False, train_encoder=True,
                 device='cuda'):
        '''
            Initialize the model

            Parameters:
                model_name (str): the name of the pretrained encoder model to use
                sense_classes (int): the number of classes for the senses classifier (not used in current implementation)
                role_classes (int): the number of classes for the role classifier
                combine_method (str): the method to combine the hidden states of the words and relations, can be 'mean', 'concatenation', 'gating' or 'gating_transform'
                norm_layer (bool): whether to use a normalization layer after the combination
                proj_dim (int): the size of the hidden states after the projection
                relation_proj (bool): whether to project the hidden states before the relational classifier
                role_layers (list): the sizes of the hidden layers of the role classifier
                role_LSTM (bool): whether to use an LSTM for the role classification (note: role_layers will be ignored, only its length is used)
                train_encoder (bool): whether to train the encoder model
                device (str): the device to use for the model
        
        '''
        super(SRL_MODEL, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        # Freeze the encoder if needed
        if not train_encoder:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.combine_method = combine_method  # 'mean', 'concatenation', 'gating'
        hidden_size = self.bert.config.hidden_size
        role_size = hidden_size
        
        # senses_layers = [(self.bert.config.hidden_size, math.ceil(sense_classes/4)), 
        #                  (math.ceil(sense_classes/4), math.ceil(sense_classes/2)),
        #                  (math.floor(sense_classes/2), sense_classes)]
        # self.senses_classifier_layers = []
        # for i, (in_features, out_features) in enumerate(senses_layers):
        #     self.senses_classifier_layers.append(nn.Linear(in_features, out_features))
        #     if i < len(senses_layers) - 1:
        #         self.senses_classifier_layers.append(nn.ReLU())
        # self.senses_classifier = nn.Sequential(*self.senses_classifier_layers)

        # Initialize the modules for the dimensionality reduction if needed
        self.dim_reduction = (proj_dim > 0)
        if self.dim_reduction:
            role_size = proj_dim
            self.rel_reduction = nn.Sequential(nn.Linear(hidden_size, proj_dim), nn.ReLU())
            self.word_reduction = nn.Sequential(nn.Linear(hidden_size, proj_dim), nn.ReLU())

        # Initialize the module for the relational classifier
        self.rel_class_reduction = relation_proj
        if self.rel_class_reduction and proj_dim>0:
            self.relational_classifier = nn.Linear(proj_dim, 1)
        else:
            self.relational_classifier = nn.Linear(hidden_size, 1)

        # Initialize the module to combine the hidden states
        if combine_method.startswith('gating'):
            self.combiner = GatedCombination(role_size, transform=(combine_method == 'gating_transform'))

        # Configure input size based on combination method
        if combine_method == 'concatenation':
            role_classifer_input_size = 2 * role_size
        else: 
            role_classifer_input_size = role_size

        # Set up the normalization layer
        self.norm = norm_layer
        if norm_layer:
            # Instantiate a LayerNorm layer
            self.norm_layer = nn.LayerNorm(role_classifer_input_size)

        # Initialize the module for the role classifier
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
            self.role_classifier = LSTMClassifier(role_classifer_input_size, role_classes, max(1,len(role_layers)), role_classes)

        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Move the model to the device
        self.to(device)
        self.device = device

        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def rel_compute(self, hidden_states, word_ids):
        """
            Compute the logits for the relational classifier

            Parameters:
                hidden_states (torch.Tensor): the hidden states of the encoder
                word_ids (list): the ids of the words in the sentence

            Returns:
                relational_logits (torch.Tensor): the logits for the relational classifier
        """
        # Apply the reduction if needed
        if self.rel_class_reduction and self.dim_reduction:
            hidden_states = self.rel_reduction(hidden_states)
            
        batch_size, seq_len, hidden_size = hidden_states.size()

        # Extract the first hidden state for each word
        first_hidden_states = torch.zeros((batch_size, max(word_ids)+1, hidden_size)).to(hidden_states.device)
        seen_words = set()
        for j, word_id in enumerate(word_ids):
            if word_id != -1 and word_id not in seen_words:
                first_hidden_states[0, word_id] = hidden_states[0, j]
                seen_words.add(word_id)

        # Compute the logits
        relational_logits = self.relational_classifier(first_hidden_states).squeeze(-1)
        return relational_logits

    def sense_compute(self, hidden_states, relations_positions):
        """
            Compute the logits for the senses classifier

            Parameters:
                hidden_states (torch.Tensor): the hidden states of the encoder
                relations_positions (list): the positions of the relations in the hidden states

            Returns:
                senses_logits (torch.Tensor): the logits for the senses classifier
        """
        phrase_indices = [[i]*len(pos) for i, pos in enumerate(relations_positions)]
        phrase_indices = [i for sublist in phrase_indices for i in sublist]
        relations_indices = [pos for positions in relations_positions for pos in positions]
        
        senses_logits = self.senses_classifier(hidden_states[phrase_indices, relations_indices])
        return senses_logits
    
    def role_compute(self, hidden_states, relations_positions, word_ids):
        """
            Compute the logits for the role classifier

            Parameters:
                hidden_states (torch.Tensor): the hidden states of the encoder
                relations_positions (list): the positions of the relations in the hidden states
                word_ids (list): the ids of the words in the sentence

            Returns:
                results (list): the logits for the role classifier
        """
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
        for i in range(batch_size):
            relation_hidden_states = []
            for pos in relations_positions[i]:
                if pos is not None and pos < seq_len:
                    # get the hidden state of the relation and apply the reduction if needed
                    relation_hidden_state = hidden_states[i, pos]
                    relation_hidden_state = self.rel_reduction(relation_hidden_state) if self.dim_reduction>0 else relation_hidden_state
                    
                    # get the hidden states of the words and apply the reduction if needed
                    word_hidden_states = first_hidden_states[i, [word_id for word_id in set(word_ids[i]) if word_id != -1]]
                    word_hidden_states = self.word_reduction(word_hidden_states) if self.dim_reduction>0 else word_hidden_states

                    # Combine the hidden states based on the method
                    if(self.combine_method == 'mean'):
                        # mean between the two states
                        combined_states = (word_hidden_states + relation_hidden_state) / 2
                    elif(self.combine_method == 'concatenation'):
                        combined_states = torch.cat([relation_hidden_state.expand_as(word_hidden_states), word_hidden_states], dim=-1)
                    elif(self.combine_method == 'gating' or self.combine_method == 'gating_transform'):
                        combined_states = self.combiner(relation_hidden_state, word_hidden_states)
                    
                    relation_hidden_states.append(combined_states)

            # Compute the logits for the role classifier
            if relation_hidden_states:
                relation_hidden_states = torch.stack(relation_hidden_states)
                relation_hidden_states = self.norm_layer(relation_hidden_states) if self.norm else relation_hidden_states

                role_logits = self.role_classifier(relation_hidden_states)
                
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
        relational_class_input = self.rel_reduction(hidden_states) if (self.dim_reduction and self.rel_class_reduction) else hidden_states
        relational_logits = self.relational_classifier(relational_class_input).squeeze(-1)
        
        # Compute the logits for the senses classifier
        # senses_logits = self.sense_compute(hidden_states, relations_positions)
        senses_logits = torch.zeros(1)
        
        # Compute the logits for the role classifier
        results = self.role_compute(hidden_states, relations_positions, word_ids)
        
        return relational_logits, senses_logits, results
    
    def inference(self,text):
        """
            Perform inference on a text

            Parameters:
                text (str): the text to perform inference on

            Returns:
                relational_logits (torch.Tensor): the logits for the relational classifier
                senses_logits (torch.Tensor): the logits for the senses classifier
                results (list): the logits for the role classifier
        """
        self.eval()

        # tokenize the text as in the training (TreebankWordTokenizer)
        tokenizer = TreebankWordTokenizer()
        text = tokenizer.tokenize(text)
        text = " ".join(text)

        with torch.no_grad():
            tokenized_text = self.tokenizer(text, return_tensors='pt')
            # if the sequence is bigger than the maximum allowed, truncate it
            if tokenized_text['input_ids'].size(1) > self.bert.config.max_position_embeddings:
                tokenized_text['input_ids'] = tokenized_text['input_ids'][:, :self.bert.config.max_position_embeddings]
                tokenized_text['attention_mask'] = tokenized_text['attention_mask'][:, :self.bert.config.max_position_embeddings]
                print("Warning: the input sequence is too long and has been truncated to the maximum allowed length")
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
            senses_logits = torch.zeros(1)

            results = self.role_compute(hidden_states, [relation_positions], [word_ids])
            # results = [result.squeeze(0) for result in results]

        return  relational_logits, senses_logits, results
