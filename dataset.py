import argparse
import numpy as npy

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import tv_tensors, datasets, models
from torchvision.transforms import v2
from torchvision.datasets import wrap_dataset_for_transforms_v2
from torchvision.datasets.voc import VOCDetection
from torchvision.ops import box_convert
from torchvision.tv_tensors import BoundingBoxes


def get_cell_location(boxes, index, cell_size=64, num_grid=7):
    cx, cy = boxes[index, :2]
    center_row = (cx / cell_size).long().clamp(0, num_grid - 1)
    center_col = (cy / cell_size).long().clamp(0, num_grid - 1)
    return (center_row, center_col)


def get_one_hot(class_index, num_classes=20):
    one_hot = torch.zeros(num_classes)
    one_hot[class_index] = 1.0
    return one_hot


def normalize_coordinate(x, y, w, h, image_size=(448, 448), cell_size=64):
    norm_x = (x % cell_size) / cell_size
    norm_y = (y % cell_size) / cell_size
    norm_w, norm_h = w / image_size[0], h / image_size[1]
    result = torch.tensor([norm_x, norm_y, norm_w, norm_h], dtype=torch.float32)
    return result


def get_normalized_coordinate(boxes, index):
    return normalize_coordinate(boxes[index, 0], boxes[index, 1], boxes[index, 2], boxes[index, 3])


def transform_to_yolo_target(
        boxes: BoundingBoxes, labels: Tensor, num_box=2, num_grid=7, num_classes=20
) -> Tensor:

    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0)
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
    box_convert(boxes, in_fmt='xyxy', out_fmt='cxcywh')

    ground_truth = torch.zeros(num_grid, num_grid, num_classes + 5 * num_box)
    existing_boxes =npy.zeros(shape=[7, 7], dtype=npy.int32)
    for i in range(len(boxes)):
        cell_x, cell_y = get_cell_location(boxes, i)
        if existing_boxes[cell_x, cell_y] < num_box:

            ground_truth[cell_x, cell_y, :num_classes] = get_one_hot(labels[i] - 1)
            ground_truth[cell_x, cell_y, num_classes] = 1
            start_index = num_classes + 1 + existing_boxes[cell_x, cell_y] * 5
            ground_truth[cell_x, cell_y, start_index : start_index + 4] = get_normalized_coordinate(boxes, i)

            existing_boxes[cell_x, cell_y] += 1

    return ground_truth


def collate_fn(batch):
    images, annotations = zip(*batch)
    images = torch.stack(images)
    targets = torch.stack([annotation["target"] for annotation in annotations])
    return (images, targets)


class TransformWrapper:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        transformed_data = self.transforms(*args)
        #print("test")
        """
        BoundingBoxes is subclass of Tensor
        
        'boxes': BoundingBoxes([[174, 131, 335, 322]], format=BoundingBoxFormat.XYXY, canvas_size=(448, 448)) 
        'labels': tensor([7])
        """
        transformed_data[1]["target"] = transform_to_yolo_target(
            transformed_data[1]["boxes"], transformed_data[1]["labels"]
        )
        return transformed_data


def get_transforms(is_train=False):
    """

    See :ref:`torchvision.transforms.v2._transform.py` and `torchvision.transforms.v2._container.py` for more details.
    """
    voc_mean, voc_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if is_train:
        transforms = v2.Compose([
            v2.ToImage(),
            v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            v2.RandomAffine(degrees=0, scale=(0.8, 1.2), translate=(0.1, 0.1), shear=10),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomPerspective(distortion_scale=0.2, p=0.5),

            v2.Resize((448, 448)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=voc_mean, std=voc_std)
        ])
    else:
        transforms = v2.Compose([
            v2.ToImage(),
            v2.Resize((448, 448)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=voc_mean, std=voc_std)
        ])
    return TransformWrapper(transforms)


def get_dataloaders(config):
    datasets = [
        wrap_dataset_for_transforms_v2(
            VOCDetection(root="data", year='2007', image_set=set_type,
                         download=True, transforms=get_transforms(set_type == 'train')))
        for set_type in ['train', 'val']
    ]
    train_loader = DataLoader(dataset=datasets[0], batch_size=config.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=config.num_workers)
    test_loader = DataLoader(dataset=datasets[1], batch_size=config.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=config.num_workers)
    return (train_loader, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_workers', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Specifies the batch size of data loaders. (default: 64)")
    config = parser.parse_args()
    train_loader, test_loader = get_dataloaders(config)
    print(len(train_loader), len(test_loader))
    for i in range(30):
        print(train_loader.dataset[i][0].shape, train_loader.dataset[i][1]['target'])
