import sys
sys.path.append('.')
from dataloaders.UP_dataloader import UP_Dataset, roles
from torch.utils.data import DataLoader
import torch

senses = []

def collate_fn(batch):
    global senses
    # breakpoint()
    max_len = max(len(item['tokenized_text']) for item in batch)
    
    input_ids = []
    word_ids = []
    attention_masks = []
    role_labels = []
    rel_senses = []
    relations = []
    
    relation_label_masks = torch.zeros((len(batch), max_len))
    relation_labels = torch.zeros((len(batch), max_len))

    # should be a 1D tensor with the same length as the number of relations in the batch
    senses_labels = torch.zeros(sum(len(item['rel_position']) for item in batch), len(senses))

    relation_num = 0

    # breakpoint()
    
    for index, item in enumerate(batch):
        input_ids.append(item['tokenized_text'] + [0] * (max_len - len(item['tokenized_text'])))
        word_ids.append(item['word_ids'] + [-1] * (max_len - len(item['word_ids'])))
        attention_masks.append(item['attention_mask'] + [0] * (max_len - len(item['attention_mask'])))

        # Senses
        # for i in range(len(item['rel_position'])):
        #     try:
        #         senses_labels[relation_num][senses.index(item['labels'][i]['sense'])] = 1
        #     except:
        #         print(f"Sense not found: {item['labels'][i]['sense']}")
        #     relation_num += 1
    
        # relations
        rel_pos = []
        for i in item['rel_position']:
            # search for the first token for that relation
            pos = word_ids[-1].index(i)
            rel_pos.append(pos)
        relations.append(rel_pos)

        for rel in rel_pos:
            relation_labels[index][rel] = 1

        seen_words = set()
        for i,word in enumerate(item['word_ids']):
            if (word not in seen_words) and (word != -1):
                relation_label_masks[index][i] = 1
                seen_words.add(word)

        # SRL labels
        phrase_labels = []

        for label_set in item['labels']:
            binary_labels = [[0] * (len(roles)-2) for _ in range(len(item['text'].split()))]
            for i, label in enumerate(label_set['SRL']):
                for l in label:
                    binary_labels[i][l-2] = 1 #because we do not want to classify the relation label and none
            phrase_labels.append(torch.tensor(binary_labels))
        role_labels.append(torch.stack(phrase_labels))
        
        rel_senses.append([label_set['sense'] for label_set in item['labels']])
    
    input_ids = torch.tensor(input_ids)
    
    # breakpoint()
    return {
        'text': [item['text'] for item in batch], # For debugging purposes
        'input_ids': input_ids,
        'attention_masks': torch.tensor(attention_masks),
        'word_ids': word_ids,

        'relation_position': relations,
        'relation_label_mask': relation_label_masks,
        'relation_label': relation_labels,

        'senses': rel_senses,
        'senses_labels': senses_labels,
        
        'role_labels': role_labels,
    }
    

def get_dataloaders(root, batch_size=32, shuffle=True):
    train_dataset = UP_Dataset(root + "en_ewt-up-train.tsv", "bert-base-uncased")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)

    dev_dataset = UP_Dataset(root + "en_ewt-up-dev.tsv", "bert-base-uncased")
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)

    test_dataset = UP_Dataset(root + "en_ewt-up-test.tsv", "bert-base-uncased")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)

    global senses
    senses = train_dataset.senses 
    num_senses = len(senses)
    num_roles = len(roles) - 2

    return train_loader, dev_loader, test_loader, num_senses, num_roles