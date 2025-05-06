import os
import sys
sys.path.append('.')
sys.path.append('..')

from dataloaders.preprocess_nombank import main as preprocess_UP
from dataloaders.preprocess_UP import main as preprocess_nombank


def preprocess_datasets(UP=True, nombank=False):
    # create folder datasets/preprocessed
    if (not os.path.exists('datasets/preprocessed')):
        os.makedirs('datasets/preprocessed')
    
    if UP:
        preprocess_UP()
    if nombank:
        preprocess_nombank()

if __name__ == '__main__':
    preprocess_datasets(UP=True, nombank=False)