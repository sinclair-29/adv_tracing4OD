import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import tv_tensors, datasets, models
from torchvision.transforms import v2
from torchvision.datasets.voc import VOCDetection


def transform_to_yolo_target():
    pass

class TransformWrappe:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        transformed_data = self.transforms(*args)

        return transformed_data


def get_transforms(is_train=False):
    """

    See :ref:`torchvision.transforms.v2._transform.py` and `torchvision.transforms.v2._container.py` for more details.
    """
    voc_mean, voc_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if is_train:
        transforms = v2.Compose[
            v2.ToImage(),
            v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            v2.RandomAffine(degrees=0, scale=(0.8, 1.2), translate=(0.1, 0.1), shear=10),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomPerspective(distortion_scale=0.2, p=0.5),

            v2.Resize((448, 448)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=voc_mean, std=voc_std)
        ]
    else:
        transforms = v2.Compose[
            v2.ToImage(),
            v2.Resize((448, 448)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=voc_mean, std=voc_std)
        ]
    return transforms


def get_dataloaders(config):
    datasets = [
        VOCDetection(root="data", year='2007', image_set=set_type,
                     download=True, transforms=get_transforms(set_type == 'train'))
        for set_type in ['train', 'val']
    ]
    train_loader = DataLoader(dataset=datasets[0], batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers)
    test_loader = DataLoader(dataset=datasets[1], batch_size=config.batch_size,
                              shuffle=False, num_workers=config.num_workers)
    return (train_loader, test_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_workers', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Specifies the batch size of data loaders. (default: 64)")
    config = parser.parse_args()
    train_loader, test_loader = get_dataloaders(config)
    print(len(train_loader), len(test_loader))
