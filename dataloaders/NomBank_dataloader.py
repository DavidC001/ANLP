import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tkinter as tk
from tkinter import ttk
from torch import nn
import sys 
sys.path.append('.')

# BERT fast TOKENIZER
from transformers import AutoTokenizer

roles = ['rel', 'None', 'ARG1', 'ARG0', 'ARG2', 
         'ARGM-TMP', 'Support', 'ARGM-MNR', 'ARGM-LOC', 
         'ARG3', 'ARGM-PNC', 'ARGM-EXT', 'ARGM-NEG', 
         'ARG4', 'ARG8', 'ARGM-ADV', 'ARG9', 'ARGM-DIR', 
         'ARGM-DIS', 'ARG5', 'ARGM-CAU', 'ARGM-PRD']

class NOM_Dataset(Dataset):
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
            phrase_info, labels = line.split(' ||| ')
            phrase_info = phrase_info.split(' ')
            phrase_id = "_".join(phrase_info[:2])
            text = " ".join(phrase_info[2:])
            
            if phrase_id not in self.data:
                tokenized = tokenizer(text)
                word_ids = tokenized.word_ids()
                word_ids = [(word_id if word_id is not None else -1) for word_id in word_ids]

                # brutta soluzione a J.
                delta = 0
                seen_ids = set()
                for i, word_id in enumerate(word_ids):
                    if word_id <= 0: continue
                    start, end = tokenized.word_to_chars(word_id)
                    if text[start-1] != ' ' and word_id not in seen_ids:
                        delta += 1
                        seen_ids.add(word_id)
                    word_ids[i] = word_id - delta
                
                self.data[phrase_id] = {
                    "text": text,
                    "tokenized_text": tokenized['input_ids'],
                    "attention_mask": tokenized['attention_mask'],
                    "word_ids": word_ids,
                    "labels": [],
                    "rel_position": []
                }
            
            labels = labels.split(' ')
            sense = labels[0]
            if sense not in self.sense_count:
                self.sense_count[sense] = 0
            self.senses.add(sense)
            self.sense_count[sense] += 1

            SRL = labels[1:]
            SRL_labels = [[]] * len(text.split())
            rel_position = None
            for label in SRL:
                label = label.split('-')
                if not len(label) > 1: #sometimes there is a space before the \n that is considered as a label
                    continue

                spans = label[0].replace('\n', '')
                span_text = spans
                #split when there is a * or , in the span
                spans = spans.split('*')
                spans = [span.split(',') for span in spans]
                spans = [item for sublist in spans for item in sublist]

                label = '-'.join(label[1:]).replace('\n', '').replace('-REF', '')
                if "-" in label and not label.startswith("ARGM"):
                    #split into multiple labels
                    label = label.split('-')
                    for l in label[1:]:
                        SRL.append(f"{span_text}-ARGM-{l}")
                    label = label[0]

                added_rel = False
                for span in spans:
                    self.SRLs.add(label)
                    start, end = span.split(':')
                    start, end = int(start.split("_")[0]), int(end.split("_")[0])+1 # _ when it's a subword labeling, we only care about the first part
                    # Handle multiple roles for the same span
                    for i in range(start, end):
                        SRL_labels[i] = SRL_labels[i] + [roles.index(label)]

                    if label == 'rel' and not added_rel:
                        added_rel = True
                        rel_position = start
            
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
                labels_str = '\n\t\t\t'.join([roles[label] for label in labels_set['SRL'][j]])
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

    dataset = NOM_Dataset("datasets/preprocessed/train.srl", "bert-base-uncased")
    # breakpoint()

    create_gui(dataset)