import time
import logging
import argparse

from .models import yolov1
from dataset import

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s ",
    datefmt="%a %d %b %Y %H:%M:%S"
)


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


def train():
    pass


def main():
    config = parse_arguments()
    dataset =

if __name__ == "__main__":
    main()