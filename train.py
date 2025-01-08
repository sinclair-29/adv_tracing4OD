import time
import argparse

import torch

from .models import yolov1
from dataset import get_dataloaders


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
    parser.add_argument('--mask_dimension', type=int, default=100,
                        help='Specifies the dimensions of the mask. (default: 100)')

    return parser.parse_args()


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'



def main():
    config = parse_arguments()
    train_loader, test_loader = get_dataloaders(config)

if __name__ == "__main__":
    main()