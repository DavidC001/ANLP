from dataloaders.preprocess_nombank import main as preprocess_UP
from dataloaders.preprocess_UP import main as preprocess_nombank

def preprocess_datasets(UP=True, nombank=True):
    if UP:
        preprocess_UP()
    if nombank:
        preprocess_nombank()