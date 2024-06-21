from dataloaders.preprocess_nombank import main as preprocess_UP
from dataloaders.preprocess_UP import main as preprocess_nombank
import os

def preprocess_datasets(UP=True, nombank=True):
    # create folder datasets/preprocessed
    if (not os.path.exists('datasets/preprocessed')):
        os.makedirs('datasets/preprocessed')
    
    if UP:
        preprocess_UP()
    if nombank:
        preprocess_nombank()