from typing import List, Tuple, Optional

import torch
from torchmetrics import Metric

from vlmeval.dataset.utils.bigdocs.metrics_utils import bootstrap_std


class TripletF1Metric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_f1", default=list(), dist_reduce_fx="cat")

    def update(self,
               prediction: Optional[List[Tuple[str, Optional[str], str]]],
               reference: Optional[List[Tuple[str, Optional[str], str]]]):
        if prediction is not None and reference is not None:
            f1 = calculate_triplet_f1_score(prediction, reference)
            self.total_f1.append(f1)
        else:
            self.total_f1.append(0.0)

    def compute(self):
        if len(self.total_f1) == 0:
            return {"triplet_f1_mean": torch.tensor(0.0), "triplet_f1_std": torch.tensor(0.0)}
        std, lb, ub = bootstrap_std(torch.tensor(self.total_f1), n_bootstrap=1000, ci=0.95)
        return {"triplet_f1_mean": torch.tensor(self.total_f1).mean(), "triplet_f1_std": std}


def calculate_triplet_f1_score(prediction: List[Tuple[str, Optional[str], str]],
                               reference: List[Tuple[str, Optional[str], str]]) -> float:
    prediction_set = set(prediction)
    reference_set = set(reference)

    true_positives = len(reference_set & prediction_set)
    false_positives = len(prediction_set - reference_set)
    false_negatives = len(reference_set - prediction_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    if precision + recall == 0:
        return 0
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1
