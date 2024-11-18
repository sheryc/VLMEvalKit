from typing import Union, Sequence, Dict

import torch
from evaluate import load
from torchmetrics import Metric
from transformers import AutoTokenizer

from vlmeval.dataset.utils.bigdocs.metrics_utils import bootstrap_std

bert_score = load("bertscore")


class BERTScoreMetric(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(kwargs['model_name_or_path'], use_fast=True,
                                                       clean_up_tokenization_spaces=False)
        self.add_state("total_preds", default=list(), dist_reduce_fx="cat")
        self.add_state("total_targets", default=list(), dist_reduce_fx="cat")

    def update(self, preds: Union[str, Sequence[str]], target: Union[str, Sequence[str]]) -> None:
        """Store predictions/references for computing BERT scores.

        It is necessary to store sentences in a tokenized form to ensure the DDP mode working.

        """
        if isinstance(preds, list):
            preds = preds[0]
        if isinstance(target, list):
            target = target[0]
        preds = self.tokenizer.decode(
            self.tokenizer(preds, add_special_tokens=False, verbose=False)['input_ids'][:510])
        target = self.tokenizer.decode(
            self.tokenizer(target, add_special_tokens=False, verbose=False)['input_ids'][:510])

        self.total_preds.append(preds)
        self.total_targets.append(target)

    def compute(self) -> Dict[str, torch.Tensor]:
        if len(self.total_preds) == 0:
            return {"bertscoref1_mean": torch.tensor(0.0), "bertscoref1_std": torch.tensor(0.0)}
        f1 = torch.tensor(
            bert_score.compute(predictions=self.total_preds, references=self.total_targets, lang='en')['f1'])
        std, lb, ub = bootstrap_std(f1, n_bootstrap=1000, ci=0.95)
        return {"bertscoref1_mean": f1.mean(), "bertscoref1_std": std}
