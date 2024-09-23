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
        if transform:
            self.transform_layer = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, relation_hidden_state, word_hidden_states):
        combined = torch.cat(
            [relation_hidden_state.expand_as(word_hidden_states), word_hidden_states],
            dim=-1,
        )
        gating_scores = torch.sigmoid(self.gate(combined))

        if self.transform:
            transformed = torch.tanh(self.transform_layer(combined))
        else:
            transformed = relation_hidden_state.expand_as(word_hidden_states)

        return gating_scores * transformed + (1 - gating_scores) * word_hidden_states


class RNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, type="LSTM"):
        """
        Initialize the RNN layer

        Parameters:
            input_size (int): the size of the input
            hidden_size (int): the size of the hidden states
            num_layers (int): the number of layers
            type (str): the type of RNN to use, can be 'LSTM', 'GRU' or 'RNN'
        """
        super(RNNLayer, self).__init__()
        models = {"LSTM": nn.LSTM, "GRU": nn.GRU, "RNN": nn.RNN}
        self.rnn = models[type](
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):

        # Forward propagate RNN
        out, _ = self.rnn(
            x
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # add forward and backward
        shape = out.shape
        out = out.view(shape[0], shape[1], 2, -1)
        out = torch.sum(out, dim=2)
        # Normalize
        out = self.norm(out)
        # Residual connection
        out = out + x

        return out


class CombinationModule(nn.Module):
    def __init__(self, combine_method, role_size, FFN_layers, num_heads, norm_layer):
        """initialize combination module

        Args:
            combine_method (str): combination strategy
            role_size (int): size of input embeddings
            FFN_layers (int): number of layers in the FFN for the combined representation
            num_heads (int): number of head for the multiHeadAttention combination strategy
            norm_layer (bool): whether to apply layer normalization
        """
        super(CombinationModule, self).__init__()

        assert combine_method in [
            "gating",
            "soft_attention",
            "multiHeadAttention",
            "mean",
            "concatenation",
        ]

        if combine_method == "concatenation":
            self.out_dim = 2 * role_size
        else:
            self.out_dim = role_size

        # Initialize the module to combine the hidden states
        if combine_method == "gating" or combine_method == "soft_attention":
            self.gate = GatedCombination(
                role_size, transform=(combine_method == "gating")
            )

        if combine_method == "multiHeadAttention":
            self.gate = GatedCombination(role_size, transform=False)
            self.mult_att = nn.MultiheadAttention(
                embed_dim=role_size, num_heads=num_heads, batch_first=True
            )
            
        FFN_layers_modules = []
        for i in range(FFN_layers):
            FFN_layers_modules.append(nn.Linear(role_size, role_size))
            FFN_layers_modules.append(nn.GELU())
        self.FFN = nn.Sequential(*FFN_layers_modules)
        
        self.norm = norm_layer
        if norm_layer:
            self.norm_layer = nn.LayerNorm(self.out_dim)

        self.combine_method = combine_method

    def forward(self, relation, words):
        """combine the hidden states
        Returns:
            torch.Tensor: combined states
        """
        if self.combine_method == "mean":
            # mean between the two states
            combined_states = words + relation
        elif self.combine_method == "concatenation":
            combined_states = torch.cat([relation.expand_as(words), words], dim=-1)
        elif self.combine_method == "gating" or self.combine_method == "soft_attention":
            combined_states = self.gate(relation, words)
        else:
            values = self.gate(relation, words)
            combined_states, _ = self.mult_att(relation.expand_as(words), words, values)

        combined_states = self.FFN(combined_states)
        combined_states += words
        combined_states = self.norm_layer(combined_states) if self.norm else combined_states

        return combined_states


class SRL_MODEL(nn.Module):
    def __init__(
        self,
        model_name,
        sense_classes,
        role_classes,
        combine_method="mean",
        attention_heads=1,
        FFN_layers=2,
        norm_layer=False,
        proj_dim=0,
        relation_proj=False,
        role_RNN=False,
        RNN_type="LSTM",
        train_encoder=True,
        train_embedding_layer=True,
        dropout_prob=0,
        device="cuda",
    ):
        """
        Initialize the model

        Parameters:
            model_name (str): the name of the pretrained encoder model to use
            sense_classes (int): the number of classes for the senses classifier (not used in current implementation)
            role_classes (int): the number of classes for the role classifier
            combine_method (str): the method to combine the hidden states of the words and relations, can be 'mean', 'concatenation', 'gating' or 'soft_attention'
            attention_heads (int): the number of attention heads, if multiHeadAttention is used
            FFN_layers (int): the number of layers in the FFN for the combined representation
            norm_layer (bool): whether to use a normalization layer after the combination
            proj_dim (int): the size of the hidden states after the projection
            relation_proj (bool): whether to project the hidden states before the relational classifier
            role_RNN (bool): whether to use a recurrent network for the role classification
            RNN_type (str): the type of RNN to use in the role classifier
            train_encoder (bool): whether to train the encoder model
            train_embedding_layer (bool): whether to train the embedding layer of the encoder model
            device (str): the device to use for the model

        """
        super(SRL_MODEL, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        # do not train embedding layers
        if not train_embedding_layer:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False

        # breakpoint()

        # Freeze the encoder if needed
        if not train_encoder:
            for param in self.bert.parameters():
                param.requires_grad = False

        hidden_size = self.bert.config.hidden_size
        role_size = hidden_size

        # senses_layers = [(self.bert.config.hidden_size, math.ceil(sense_classes/4)),
        #                  (math.ceil(sense_classes/4), math.ceil(sense_classes/2)),
        #                  (math.floor(sense_classes/2), sense_classes)]
        # self.senses_classifier_layers = []
        # for i, (in_features, out_features) in enumerate(senses_layers):
        #     self.senses_classifier_layers.append(nn.Linear(in_features, out_features))
        #     if i < len(senses_layers) - 1:
        #         self.senses_classifier_layers.append(nn.GELU())
        # self.senses_classifier = nn.Sequential(*self.senses_classifier_layers)

        # Initialize the modules for the dimensionality reduction if needed
        self.dim_reduction = proj_dim > 0
        if self.dim_reduction:
            role_size = proj_dim
            self.rel_reduction = nn.Sequential(
                nn.Linear(hidden_size, proj_dim), nn.GELU()
            )
            self.linear_rel_reduction = nn.Linear(hidden_size, proj_dim)
            self.rel_reduction_norm = nn.LayerNorm(proj_dim)
            self.word_reduction = nn.Sequential(
                nn.Linear(hidden_size, proj_dim), nn.GELU()
            )
            self.linear_word_reduction = nn.Linear(hidden_size, proj_dim)
            self.word_reduction_norm = nn.LayerNorm(proj_dim)

        # Initialize the module for the relational classifier
        self.rel_class_reduction = relation_proj
        if self.rel_class_reduction and proj_dim > 0:
            self.relational_classifier = nn.Linear(proj_dim, 1)
        else:
            self.relational_classifier = nn.Linear(hidden_size, 1)

        self.combiner = CombinationModule(combine_method, role_size, FFN_layers, attention_heads, norm_layer)

        # Set up the normalization layer
        self.norm = norm_layer

        # dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Initialize the module for the role classifier
        self.role_RNN = role_RNN
        if role_RNN:
            self.RNN_layer = RNNLayer(hidden_size, hidden_size, 1, RNN_type)

        self.role_classifier = nn.Linear(self.combiner.out_dim, role_classes)

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
            hidden_states = self.rel_reduction(
                hidden_states
            ) + self.linear_rel_reduction(hidden_states)
            hidden_states = self.rel_reduction_norm(hidden_states)

        batch_size, seq_len, hidden_size = hidden_states.size()

        # Extract the first hidden state for each word
        first_hidden_states = torch.zeros(
            (batch_size, max(word_ids) + 1, hidden_size)
        ).to(hidden_states.device)
        seen_words = set()
        for j, word_id in enumerate(word_ids):
            if word_id != -1 and word_id not in seen_words:
                first_hidden_states[0, word_id] = hidden_states[0, j]
                seen_words.add(word_id)

        # Compute the logits
        first_hidden_states = self.dropout(first_hidden_states)
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
        phrase_indices = [[i] * len(pos) for i, pos in enumerate(relations_positions)]
        phrase_indices = [i for sublist in phrase_indices for i in sublist]
        relations_indices = [
            pos for positions in relations_positions for pos in positions
        ]

        senses_logits = self.senses_classifier(
            hidden_states[phrase_indices, relations_indices]
        )
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

        if self.role_RNN:
            hidden_states = self.RNN_layer(hidden_states)

        # Extract the first hidden state for each word
        first_hidden_states = torch.zeros(
            (batch_size, max([max(words_id) for words_id in word_ids]) + 1, hidden_size)
        ).to(hidden_states.device)
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
                    if self.dim_reduction > 0:
                        relation_hidden_state = self.rel_reduction(
                            relation_hidden_state
                        ) + self.linear_rel_reduction(relation_hidden_state)
                        relation_hidden_state = self.rel_reduction_norm(
                            relation_hidden_state
                        )
                    # relation_hidden_state = self.dropout(relation_hidden_state)

                    # get the hidden states of the words and apply the reduction if needed
                    word_hidden_states = first_hidden_states[
                        i, [word_id for word_id in set(word_ids[i]) if word_id != -1]
                    ]
                    if self.dim_reduction > 0:
                        word_hidden_states = self.word_reduction(
                            word_hidden_states
                        ) + self.linear_word_reduction(word_hidden_states)
                        word_hidden_states = self.word_reduction_norm(
                            word_hidden_states
                        )
                    # word_hidden_states = self.dropout(word_hidden_states)

                    # Combine the hidden states based on the method
                    combined_states = self.combiner(
                        relation_hidden_state, word_hidden_states
                    )

                    relation_hidden_states.append(combined_states)

            # Compute the logits for the role classifier
            if relation_hidden_states:
                relation_hidden_states = torch.stack(relation_hidden_states)

                relation_hidden_states = self.dropout(relation_hidden_states)
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
        hidden_states = self.dropout(hidden_states)

        # Compute the logits for the relational classifier
        relational_class_input = (
            self.rel_reduction(hidden_states)
            if (self.dim_reduction and self.rel_class_reduction)
            else hidden_states
        )
        relational_class_input = self.dropout(relational_class_input)
        relational_logits = self.relational_classifier(relational_class_input).squeeze(
            -1
        )

        # Compute the logits for the senses classifier
        # senses_logits = self.sense_compute(hidden_states, relations_positions)
        senses_logits = torch.zeros(1)

        # Compute the logits for the role classifier
        results = self.role_compute(hidden_states, relations_positions, word_ids)

        return relational_logits, senses_logits, results

    def inference(self, text):
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
            tokenized_text = self.tokenizer(text, return_tensors="pt")
            # if the sequence is bigger than the maximum allowed, truncate it
            if (
                tokenized_text["input_ids"].size(1)
                > self.bert.config.max_position_embeddings
            ):
                # print red
                print(
                    "\033[91m"
                    + "Warning: the input sequence is too long and has been truncated to the maximum allowed length"
                    + "\033[0m"
                )
                return None, None, None
            input_ids = tokenized_text["input_ids"]
            attention_mask = tokenized_text["attention_mask"]
            word_ids = tokenized_text.word_ids()
            word_ids = [
                (word_id if word_id is not None else -1) for word_id in word_ids
            ]
            # brutta soluzione a J.
            delta = 0
            seen_ids = set()
            for i, word_id in enumerate(word_ids):
                if word_id <= 0:
                    continue
                start, end = tokenized_text.word_to_chars(word_id)
                if text[start - 1] != " " and word_id not in seen_ids:
                    delta += 1
                    seen_ids.add(word_id)
                word_ids[i] = word_id - delta

            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state

            relational_logits = self.rel_compute(hidden_states, word_ids).squeeze(0)
            # apply sigmoid
            relational_probabilities = torch.sigmoid(relational_logits)
            relation_positions = [
                i
                for i in range(len(relational_probabilities))
                if relational_probabilities[i] > 0.75
            ]
            relation_positions = [word_ids.index(i) for i in relation_positions]

            # senses_logits = self.sense_compute(hidden_states, [relation_positions])
            senses_logits = torch.zeros(1)

            results = self.role_compute(hidden_states, [relation_positions], [word_ids])
            # results = [result.squeeze(0) for result in results]

        return relational_logits, senses_logits, results
