import os
import argparse
import sys
import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms as T
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchmetrics.functional import accuracy


def get_args(stdin):
    """
    Retrieve input arguments.
    """
    parser = argparse.ArgumentParser(stdin)
    parser.add_argument('--batch_size', type=int, default=256, help='The batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='The learning rate.')
    parser.add_argument('--gpus', type=int, default=1, help='Number of gpus to use.')
    parser.add_argument('--num_workers', type=int, default=8, help='The number of workers.')
    parser.add_argument('--epochs', type=int, default=10, help='The number of epochs.')
    parser.add_argument('--nodes', type=int, default=1, help='The number of nodes.')
    args = parser.parse_args()
    pprint.pprint(vars(args), indent=4)
    return args

class TinyNet(pl.LightningModule):
    """
    Model definition.
    """
    def __init__(self, args, in_dim=784, h_dim=1024, num_classes=10, dropout=0.1):
        super().__init__()
        self.args = args
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h_dim, num_classes)
        )

    def forward(self, x):
        return self.mlp(torch.flatten(x, 1)) # [B, N_CL]

    def training_step(self, batch, i, mode='train'):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(F.softmax(logits, 1), y)
        self.log(f'{mode}_loss', loss, prog_bar=True)
        self.log(f'{mode}_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, i):
        return self.training_step(batch, i, mode='val')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr)

def main(args):
    """
    Main function.
    """

    # Dataloader
    ds_args = {'root': './data', 'download': True, 'transform': T.ToTensor()}
    dl_train = DataLoader(MNIST(train=True, **ds_args), batch_size=args.batch_size, shuffle=True)
    dl_val = DataLoader(MNIST(train=False, **ds_args), batch_size=args.batch_size)

    # Model
    model = TinyNet(args)

    # Trainer
    trainer = pl.Trainer(gpus=args.gpus, max_epochs=args.epochs, num_nodes=args.nodes, logger=False, 
                         enable_checkpointing=False)
    trainer.fit(model, train_dataloaders=dl_train, val_dataloaders=dl_val)


if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    main(args)
