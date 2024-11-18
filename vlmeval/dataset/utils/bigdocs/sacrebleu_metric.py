import torch
from torchmetrics import Metric
from torchmetrics.text import SacreBLEUScore

from vlmeval.dataset.utils.bigdocs.metrics_utils import bootstrap_std


class SacreBLEUMetric(Metric):
    def __init__(self):
        super().__init__()
        self.sacrebleu = SacreBLEUScore()
        self.add_state("total_score", default=list(), dist_reduce_fx="cat")

    def update(self, target: str, preds: str):
        if len(target) > 0 and len(preds) > 0:
            score = self.sacrebleu([preds], [[target]])
            self.total_score.append(score)
        else:
            self.total_score.append(0.0)

    def compute(self):
        if len(self.total_score) == 0:
            return {"sacrebleu_mean": torch.tensor(0.0), "sacrebleu_std": torch.tensor(0.0)}
        std, lb, ub = bootstrap_std(torch.tensor(self.total_score), n_bootstrap=1000, ci=0.95)
        return {"sacrebleu_mean": torch.tensor(self.total_score).mean(), "sacrebleu_std": std}
