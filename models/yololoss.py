import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops import box_convert, box_iou


def convert_to_xyxy_format(bboxes, image_size=(448, 448)):
    """
    Converts the bounding boxes from normalized yolo format to xyxy format

    Args:
        bboxes (Tensor([batch_size, 7, 7, 2, 4]))
    Returns:
        bboxes (Tensor([batch_size, 7, 7, 4])) with xyxy format
    """
    batch_size, num_grid, _, num_boxes, _ = bboxes.size()
    # assume the weight of images equals to the height
    cell_size = image_size[0] / num_grid
    # norm_x Tensor([batch_size, 7, 7, 2, 1])
    norm_x, norm_y, norm_w, norm_h = bboxes.unbind(dim=-1)

    x_grid_indices, y_grid_indices = torch.meshgrid(
        torch.arange(num_grid, device=bboxes.device),
        torch.arange(num_grid, device=bboxes.device),
        indexing="ij"
    )
    x_grid_indices = x_grid_indices[None, :, :, None, None].expand(batch_size, num_grid, num_grid, num_boxes, 1)
    y_grid_indices = y_grid_indices[None, :, :, None, None].expand(batch_size, num_grid, num_grid, num_boxes, 1)
    denorm_x = cell_size * (norm_x + x_grid_indices)
    denorm_y = cell_size * (norm_y + y_grid_indices)
    denorm_w, denorm_h = norm_w * image_size[0], norm_h * image_size[1]

    # xyxy_format tensor[, 4]
    # xyxy_format size:
    # pred: [batch_size, 7, 7, 2, 4]
    # ground-truth: [batch_size, 7, 7, 1, 4]
    xyxy_format = box_convert(
        torch.cat((denorm_x, denorm_y, denorm_w, denorm_h), dim=-1),
        in_fmt="cxcywh", out_fmt="xyxy"
    )

    zero_mask = (bboxes == 0).all(dim=-1, keepdim=True)
    xyxy_format = xyxy_format * (~zero_mask)
    return xyxy_format.squeeze(-2)


def get_responsible_bbox(pred, ground_truth, num_grid=7, num_box=2):
    """

    Args:
        pred (Tensor([batch_size, 7, 7, 2, 4]))
        ground_truth (Tensor([batch_size, 7, 7, 4]))
    Returns:

    """
    batch_size = pred.size(dim=0)

    pred_coordinates = convert_to_xyxy_format(pred)
    ground_truth_coordinates = convert_to_xyxy_format(ground_truth.unsqueeze(-2))
    num_bbox = batch_size * num_grid * num_grid
    # ious Tensor(num_bbox, num_bbox, 2)
    ious = (
        box_iou(pred_coordinates.view(-1, 4),  ground_truth_coordinates.view(-1, 4))
    )

    return (responsible_indices, ious)


class YoloLoss(nn.Module):
    lambda_coord = 5
    lambda_noobj = 0.5

    def __init__(self, num_classes=20):
        super().__init__()

    def forword(
            self, pred: Tensor, grond_truth: Tensor
    ) -> Tensor:
        """
        Args:
            predict: (batch_size, 7, 7, 30) [one-hot class, confidence, x, y, w, h, confidence, x, y, w, h, ]
            grond_truth: (batch_size, 7, 7, 25) [one-hot class, 1, x, y, w, h]

        """
        num_classes = 20
        num_cell = 7
        num_box = 2
        object_mask = grond_truth[..., num_classes] == 1
        noobj_mask = ~object_mask

        pred_boxes = pred[..., num_classes:].view(-1, num_cell, num_cell, num_box, 5)
        groud_truth_boxes = grond_truth[..., num_classes:].view(-1, num_cell, num_cell, 5)
        obj_ij_mask = None,
        get_responsible_bbox(pred_boxes[..., 1:], groud_truth_boxes[..., 1:])

        box_center_loss = torch.sum(
            (pred[] )
        )

        box_width_height_loss = torch.sum()

        noobj_confidence_loss = torch.sum(
            (pred[noobj_mask][..., num_classes] - )
        )

        class_loss = torch.sum(
            (pred[object_mask][..., :num_classes] - grond_truth[object_mask][..., :num_classes]) ** 2
        )

        result = YoloLoss.lambda_coord * (box_center_loss + box_width_height_loss) \
               + object_confidence_loss \
               + YoloLoss.lambda_noobj * noobj_confidence_loss \
               + class_loss
        return result
