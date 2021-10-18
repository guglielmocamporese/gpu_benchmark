"""
In order to run this test, you have to install torch, torchvision and tqdm.
"""

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

def accuracy(logits, y):
    """
    Compute the accuracy between [B, N_CL] logits and [B] y.
    """
    return (F.softmax(logits, 1).max(1)[1] == y).to(torch.float32).mean()

def single_epoch(model, opt, dl, epoch, mode='train'):
    """
    Single train/validation epoch.
    """
    pbar = tqdm(total=len(dl.dataset), desc=f'*epoch {epoch}')
    loss_list, acc_list = [], []
    if mode == 'train':
        model.train()
    else:
        model.eval()
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        if mode == 'train':
            opt.zero_grad()
            logits = model(torch.flatten(x, 1))
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
        else:
            logits = model(torch.flatten(x, 1))
            loss = F.cross_entropy(logits, y)
        acc = accuracy(logits, y)
        loss_list += [loss.item()]
        acc_list += [acc.item()]
        pbar.update(x.shape[0])
        pbar.set_postfix({f'{mode} loss': f'{loss_list[-1]:.3f}', f'{mode} acc': f'{acc_list[-1]:.3f}'})
    loss, acc = torch.tensor(loss_list).mean(), torch.tensor(acc_list).mean()
    pbar.set_postfix({f'{mode} loss': f'{loss:.3f}', f'{mode} acc': f'{acc:.3f}'})
    pbar.close()
    return loss, acc

def train(model, dl_train, dl_val, epochs=10):
    """
    Fit the model.
    """
    model.to(device)
    opt = Adam(model.parameters(), lr=3e-4)
    for e in range(epochs):
        _, _ = single_epoch(model, opt, dl_train, e, mode='train')
        _, val_acc = single_epoch(model, opt, dl_val, e, mode='val')
    return val_acc


##################################################
# Main
##################################################

if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(35771)
    print(f'*device: {device}')

    # Model
    mlp = nn.Sequential(
        nn.Linear(28 * 28, 256), 
        nn.ReLU(inplace=True),
        nn.Linear(256, 10)
    )

    # Dataloader
    transform = transforms.ToTensor()
    dl_train = DataLoader(MNIST('./data', train=True, download=True, transform=transform), batch_size=256, pin_memory=True, shuffle=True)
    dl_val = DataLoader(MNIST('./data', train=False, download=True, transform=transform), batch_size=256, pin_memory=True)

    # Train
    val_acc = train(mlp, dl_train, dl_val, epochs=1)
    if val_acc > 0.9:
        print('*test succeeded!')
    else:
        print('*test not succeeded!')
