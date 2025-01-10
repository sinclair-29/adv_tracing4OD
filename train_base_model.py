import os
import time
import logging
import argparse

import torch
import torch.nn as nn

from models.yololoss import YoloLoss
from models.yolov1 import Yolov1Head, Yolov1Backbone
from dataset import get_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s ",
    datefmt="%a %d %b %Y %H:%M:%S"
)


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_workers', type=int, default=24)
    parser.add_argument('--num_copy', type=int, default=10,
                        help='Specifies the number of copy network. (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Specifies the learning rate of the optimizer. (default: 1e-3)")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Specifies the batch size of data loaders. (default: 64)")
    parser.add_argument('--epochs', type=int, default=50,
                        help='Specifies the number of epochs to train model. (default: 50)')

    return parser.parse_args()


def train(model, data_loader, criterion, optimizer, epoch, log_interval=100):
    model.train()

    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()



def main():
    config = parse_arguments()
    train_loader, test_loader = get_dataloaders(config)
    head, tail = Yolov1Head(), Yolov1Backbone()

    base_model = nn.Sequential(head, tail).to(device)
    criterion = YoloLoss()
    # Adam works better?
    optimizer = torch.optim.Adam(base_model.parameters(), lr=config.lr)

    for epoch in range(1, config.epochs + 1):
        train(base_model, train_loader, criterion, optimizer, epoch)

    save_dir = 'saved_models/yolov1'
    os.makedirs(save_dir, exist_ok=True)



if __name__ == "__main__":
    main()