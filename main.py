import sys
sys.path.append('.')
from model import SRL_BERT
from utils import get_dataloaders
from functions import train
import torch


def train_SRL():
    train_loader, val_loader, test_loader, num_senses, num_roles = get_dataloaders("datasets/preprocessed/", batch_size=128, shuffle=True)

    model = SRL_BERT("bert-base-uncased", num_senses, num_roles, [50], device='cuda', mean=True)

    train(model, train_loader, val_loader, test_loader,
          epochs=10, init_lr=0.0001, scheduler_step=2, scheduler_gamma=0.9,
          device='cuda', name='SRL_BERT_TEST')

    # Save the model
    torch.save(model.state_dict(), "models/SRL_BERT_TEST.pt")

if __name__ == '__main__':
    train_SRL()
