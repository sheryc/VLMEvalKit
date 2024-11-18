import torch
from torchmetrics import Metric
from torchmetrics.text import ROUGEScore

from vlmeval.dataset.utils.bigdocs.metrics_utils import bootstrap_std


class RougeMetric(Metric):
    def __init__(self):
        super().__init__()
        self.rouge = ROUGEScore()
        self.add_state("total_score", default=list(), dist_reduce_fx="cat")

    def update(self, reference: str, prediction: str):
        if len(reference) > 0 and len(prediction) > 0:
            rouge_f1 = self.rouge(prediction, reference)['rougeL_fmeasure']
            self.total_score.append(rouge_f1)
        else:
            self.total_score.append(0.0)

    def compute(self):
        if len(self.total_score) == 0:
            return {"rougeL_f1_mean": torch.tensor(0.0), "rougeL_f1_std": torch.tensor(0.0)}
        std, lb, ub = bootstrap_std(torch.tensor(self.total_score), n_bootstrap=1000, ci=0.95)
        return {"rougeL_f1_mean": torch.tensor(self.total_score).mean(), "rougeL_f1_std": std}
