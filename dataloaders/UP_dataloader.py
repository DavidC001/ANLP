import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tkinter as tk
from tkinter import ttk
from torch import nn
import sys 
sys.path.append('.')

from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer

# BERT fast TOKENIZER
from transformers import AutoTokenizer

roles = ['rel', 'None', 'ARGM-EXT', 'ARG5', 'ARG0', 
         'ARGM-LOC', 'ARGM-LVB', 'ARG1', 'ARGM-GOL', 
         'ARGM-REC', 'ARGM-NEG', 'ARGM-PRP', 'ARG2', 
         'ARG3', 'ARGM-DIR', 'ARGM-TMP', 'ARGM-MNR', 
         'ARGM-ADJ', 'ARGA', 'ARGM-DIS', 'ARGM-PRR', 
         'ARGM-MOD', 'ARG4', 'ARGM-ADV', 'ARGM-CAU', 
         'ARG1-DSP', 'ARGM-COM', 'ARGM-PRD', 'ARGM-CXN']

role_meanings = [
    "Predicate or relation",
    "None (no argument)",
    "Extent (ARGM-EXT) - The extent or degree of the action or property",
    "Beneficiary (ARG5) - The beneficiary of the action",
    "Agent (ARG0) - The doer of the action",
    "Location (ARGM-LOC) - Where the action occurs",
    "Light verb (ARGM-LVB) - Light verb construction",
    "Patient or theme (ARG1) - The entity affected by the action",
    "Goal (ARGM-GOL) - The endpoint or recipient of the action",
    "Recipient (ARGM-REC) - The recipient or beneficiary of the action",
    "Negation (ARGM-NEG) - Negation of the action",
    "Purpose (ARGM-PRP) - The purpose or reason for the action",
    "Instrument (ARG2) - The instrument or means by which the action is performed",
    "Starting point (ARG3) - The starting point or origin of the action",
    "Direction (ARGM-DIR) - The direction of the action",
    "Temporal (ARGM-TMP) - When the action occurs",
    "Manner (ARGM-MNR) - How the action is performed",
    "Adjectival (ARGM-ADJ) - Adjectival modification",
    "General argument A (ARGA) - An unspecified argument A",
    "Discourse (ARGM-DIS) - Discourse markers",
    "Purpose clause (ARGM-PRR) - Purpose or result clause",
    "Modal (ARGM-MOD) - Modal verbs or constructions",
    "Cause (ARG4) - The cause or reason for the action",
    "Adverbial (ARGM-ADV) - Adverbial modification",
    "Cause (ARGM-CAU) - The cause of the action",
    "Displaced ARG1 (ARG1-DSP) - Displaced or alternative ARG1",
    "Comitative (ARGM-COM) - Accompaniment or comitative",
    "Predicate (ARGM-PRD) - Secondary predication",
    "Complex connective (ARGM-CXN) - Complex connectives or constructions"
]


class UP_Dataset(Dataset):
    def __init__(self, file_path, tokenizer_name):
        global roles, senses

        self.senses = set()
        self.sense_count = {}
        self.SRLs = set()
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        assert tokenizer.is_fast, "Tokenizer must be fast"
        lines = open(file_path, encoding='utf-8').readlines()

        self.data = {}

        for line in lines:
            data = line.split('\t')
            phrase_id = int(data[1])
            if phrase_id not in self.data:
                tokenized = tokenizer(data[3])
                word_ids = tokenized.word_ids()

                # brutta soluzione a J.
                delta = 0
                seen_ids = set()
                for i, word_id in enumerate(word_ids):
                    if word_id == None or word_id==0: continue
                    start, end = tokenized.word_to_chars(word_id)
                    if data[3][start-1] != ' ' and word_id not in seen_ids:
                        delta += 1
                        seen_ids.add(word_id)
                    word_ids[i] = word_id - delta
                
                self.data[phrase_id] = {
                    "text": data[3],
                    "tokenized_text": tokenized['input_ids'],
                    "attention_mask": tokenized['attention_mask'],
                    "word_ids": word_ids,
                    "labels": [],
                    "rel_position": []
                }
            
            sense = data[4]
            if sense not in self.sense_count:
                self.sense_count[sense] = 0
            self.senses.add(sense)
            self.sense_count[sense] += 1
            SRL = data[5:]
            SRL_labels = [[1]] * len(data[3].split())
            rel_position = None
            for label in SRL:
                label = label.split('-')
                span = label[0]
                label = '-'.join(label[1:]).replace('\n', '').replace('C-', '').replace('R-', '')
                if(label == 'V'): continue
                self.SRLs.add(label)
                start, end = span.split(':')
                start, end = int(start), int(end) + 1
                # Handle multiple roles for the same span
                for i in range(start, end):
                    if SRL_labels[i][0] == 1:
                        SRL_labels[i] = [roles.index(label)]
                    else:
                        # Create a multi-label scenario
                        SRL_labels[i].append(roles.index(label))

                if label == 'rel':
                    rel_position = int(start)
            
            self.data[phrase_id]["labels"].append({
                "sense": sense,
                "SRL": SRL_labels
            })
            self.data[phrase_id]["rel_position"].append(rel_position)
        
        self.senses = list(self.senses)
        
        self.data = list(self.data.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def print_one_item(batch):
    for i in range(len(batch['text'])):
        data = {
            'text': batch['text'][i],
            'input_ids': batch['input_ids'][i],
            'word_ids': batch['word_ids'][i],
            'relation_position': batch['relation_position'][i],
            'labels': batch['labels'][i],
            'senses': batch['senses'][i]
        }
        print("Text:", data['text'])
        tokenized_text = data['text'].split()
        print("\tInput IDs:", data['input_ids'])
        print("\tWord IDs:", data['word_ids'])
        print("\tRelation positions:", data['relation_position'])

        for k,r in enumerate(data["relation_position"]):
            print(f"\tRelation sense {data['senses'][k]}")
            print("\t\tLabels:")
            # breakpoint()
            for j in range(len(data['labels'][k])):
                print(f"\t\t\t{tokenized_text[j]}: {[roles[q] for q in range(len(data['labels'][k][j])) if data['labels'][k][j][q] == 1]}")
        print("\n\n")

def create_gui(dataset):
    def on_select(event):
        w = event.widget
        index = int(w.curselection()[0])
        # value = w.get(index)
        
        selected_data = dataset[index]
        display_item(selected_data)

    def display_item(data):
        text_display.delete('1.0', tk.END)
        text_display.insert(tk.END, f"Text: {data['text']}\n\n")

        tokenized_text = data['text'].split()
        
        rel_positions = data.get('rel_position', [])
        
        for k, r in enumerate(rel_positions):
            labels_set = data.get('labels', [])[k]

            sense = labels_set['sense']
            text_display.insert(tk.END, f"\nRelation taking sense {sense}\n")
            text_display.insert(tk.END, "\tLabels:\n")
            
            for j in range(len(labels_set['SRL'])):
                labels_str = '\n\t\t\t'.join([role_meanings[label] for label in labels_set['SRL'][j]])
                token = tokenized_text[j] if j < len(tokenized_text) else '[UNKNOWN]'
                text_display.insert(tk.END, f"\t\t{j}: {token} -> \n\t\t\t{labels_str}\n")
            
        text_display.insert(tk.END, "\n\n")


    root = tk.Tk()
    root.title("Item Viewer")

    mainframe = ttk.Frame(root, padding="3 3 12 12")
    mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    listbox = tk.Listbox(mainframe, height=50)
    listbox.grid(column=1, row=1, sticky=(tk.W, tk.E, tk.N, tk.S))

    for idx, item in enumerate(dataset):
        listbox.insert(tk.END, f"Phrase {idx}")

    listbox.bind('<<ListboxSelect>>', on_select)

    global text_display
    text_display = tk.Text(mainframe, width=120, height=50)
    text_display.grid(column=2, row=1, sticky=(tk.W, tk.E, tk.N, tk.S))

    root.mainloop()

if __name__ == "__main__":
    from sklearn.metrics import precision_recall_fscore_support
    import numpy as np

    dataset = UP_Dataset("datasets/preprocessed/en_ewt-up-train.tsv", "bert-base-uncased")

    create_gui(dataset)