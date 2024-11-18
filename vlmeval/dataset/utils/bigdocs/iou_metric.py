from typing import Optional

import torch
from torchmetrics import Metric

from vlmeval.dataset.utils.bigdocs.metrics_utils import bootstrap_std, preprocess_bbox_string


class BBoxIOUMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_iou", default=list(), dist_reduce_fx="cat")

    def update(self,
               prediction: Optional[str],
               reference: Optional[str]):
        if prediction is not None and reference is not None:
            iou = calculate_iou(prediction, reference)
            self.total_iou.append(iou)
        else:
            self.total_iou.append(0.0)

    def compute(self):
        if len(self.total_iou) == 0:
            return {"bbox_iou_mean": torch.tensor(0.0), "bbox_iou_std": torch.tensor(0.0)}
        std, lb, ub = bootstrap_std(torch.tensor(self.total_iou), n_bootstrap=1000, ci=0.95)
        return {"bbox_iou_mean": torch.tensor(self.total_iou).mean(), "bbox_iou_std": std}


def calculate_iou(reference: str, prediction: str) -> float:
    box1 = preprocess_bbox_string(reference)
    box2 = preprocess_bbox_string(prediction)

    if not box1 or not box2:
        return 0.0

    # Unpack the box coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # normalize the box coordinates
    x1_1, x2_1 = min(x1_1, x2_1), max(x1_1, x2_1)
    y1_1, y2_1 = min(y1_1, y2_1), max(y1_1, y2_1)
    x1_2, x2_2 = min(x1_2, x2_2), max(x1_2, x2_2)
    y1_2, y2_2 = min(y1_2, y2_2), max(y1_2, y2_2)

    # Intersection coordinates
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    # Calculate intersection area
    width = max(0, xi2 - xi1)
    height = max(0, yi2 - yi1)
    intersection_area = width * height

    # Calculate each box area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = area1 + area2 - intersection_area

    iou = intersection_area / union_area
    return iou
