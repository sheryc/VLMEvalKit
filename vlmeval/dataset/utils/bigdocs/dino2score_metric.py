from typing import Any, Optional, Dict

import torch
import torch.nn as nn
from PIL import Image
from torchmetrics import Metric
from transformers import AutoImageProcessor, AutoModel

from vlmeval.dataset.utils.bigdocs.metrics_utils import bootstrap_std

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DINO2ScoreMetric(Metric):
    def __init__(self):
        super().__init__()
        self.model, self.processor = self.get_DINOv2_model("base")
        self.model = self.model.to(device)
        self.add_state("total_similarity", default=list(), dist_reduce_fx="cat")

    def update(self, reference: Optional[Image], prediction: Optional[Image]):
        if reference is None or prediction is None:
            self.total_similarity.append(0.0)
            return
        sim = self.calculate_DINOv2_similarity_score(reference, prediction, "cuda", self.model, self.processor)
        self.total_similarity.append(sim)

    def compute(self) -> Dict[str, torch.Tensor]:
        if len(self.total_similarity) == 0:
            return {"dino2score_mean": torch.tensor(0.0), "dino2score_std": torch.tensor(0.0)}
        similarity = torch.tensor(self.total_similarity)
        std, lb, ub = bootstrap_std(similarity, n_bootstrap=1000, ci=0.95)
        return {"dino2score_mean": similarity.mean(), "dino2score_std": std}

    def get_DINOv2_model(self, model_size):
        if model_size == "small":
            model_size = "facebook/dinov2-small"
        elif model_size == "base":
            model_size = "facebook/dinov2-base"
        elif model_size == "large":
            model_size = "facebook/dinov2-large"
        else:
            raise ValueError(f"model_size should be either 'small', 'base' or 'large', got {model_size}")
        return AutoModel.from_pretrained(model_size), AutoImageProcessor.from_pretrained(model_size)

    def process_input(self, image, processor):
        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, Image.Image):
            with torch.no_grad():
                inputs = processor(images=image, return_tensors="pt").to(device)
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1)
        elif isinstance(image, torch.Tensor):
            features = image.unsqueeze(0) if image.dim() == 1 else image
        else:
            raise ValueError("Input must be a file path, PIL Image, or tensor of features")
        return features

    def calculate_DINOv2_similarity_score(self, image1, image2, device, model, processor):
        model = model.to(device)

        features1 = self.process_input(image1, processor)
        features2 = self.process_input(image2, processor)

        cos = nn.CosineSimilarity(dim=1)
        sim = cos(features1, features2).item()
        sim = (sim + 1) / 2

        return sim


if __name__ == "__main__":
    pass
