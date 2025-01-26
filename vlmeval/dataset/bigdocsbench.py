from functools import partial

import pandas as pd
import torch
from tqdm import tqdm

from vlmeval.dataset.utils.bigdocs import (
    BBoxIOUMetric,
    DINO2ScoreMetric,
    HTMLSimilarityMetric,
    RMSF1Metric,
    RougeMetric,
    TexBLEUMetric,
    TripletF1Metric,
)
from vlmeval.dataset.utils.bigdocs.metrics_utils import (
    extract_triplet_from_graphviz_pydot,
    extract_triplet_from_json,
    preprocess_latex_table,
    preprocess_markdown,
    validate_plot_svg,
)
from .image_base import ImageBaseDataset
from ..smp import get_logger, load, dump


ALL_TASKS = [
    "Chart2Caption",
    "Chart2Markdown",
    "GUIVQA",
    "GUI2BBox",
    "GUI2Summary",
    "GUI2UserIntent",
    "FlowChart2Graphviz",
    "FlowChart2JSON",
    "Image2SVG",
    "Text2SVG",
    "Screenshot2HTML",
    "Table2Latex",
]


def get_pred(example):
    return example.get("prediction", "")


def get_reference(example, keep_original=False):
    reference = example.get("answer", "")
    if isinstance(reference, list):
        reference = reference[0]
    if isinstance(reference, dict):
        reference = reference.get("text", "")

    if not keep_original:
        if reference.startswith("[") and reference.endswith("]"):
            reference = reference[1:-1]
        if reference.startswith('"') and reference.endswith('"'):
            reference = reference[1:-1]
        if reference.startswith("'") and reference.endswith("'"):
            reference = reference[1:-1]
        reference = reference.replace(r"\n", "\n").replace(r"\t", "\t")
    return reference


def iterative_tensor_to_float(result_dict):
    for key in result_dict:
        if isinstance(result_dict[key], torch.Tensor):
            result_dict[key] = result_dict[key].item()
        elif isinstance(result_dict[key], dict):
            result_dict[key] = iterative_tensor_to_float(result_dict[key])
    return result_dict


def process_pred_and_reference(pred: str, reference: str, process_fn: callable):
    return process_fn(pred), process_fn(reference)


def evaluate_per_task(result, metrics_dict):
    task_name = result.get("category", "")
    assert task_name in ALL_TASKS, f"Invalid task name: {task_name}"

    if task_name == "Table2Latex":
        pred, ref = process_pred_and_reference(
            get_pred(result), get_reference(result), preprocess_latex_table
        )
        metrics_dict[task_name]["texbleu"].update(reference=ref, prediction=pred)

    elif task_name == "Chart2Markdown":
        pred, ref = process_pred_and_reference(
            get_pred(result), get_reference(result), preprocess_markdown
        )
        metrics_dict[task_name]["rmsf1"].update(reference=ref, prediction=pred)

    elif task_name in ["Text2SVG", "Image2SVG"]:
        pred, ref = process_pred_and_reference(
            get_pred(result), get_reference(result), validate_plot_svg
        )
        metrics_dict[task_name]["dino2score"].update(reference=ref, prediction=pred)

    elif task_name == "FlowChart2Graphviz":
        pred, ref = process_pred_and_reference(
            get_pred(result),
            get_reference(result),
            partial(extract_triplet_from_graphviz_pydot, use_shape=True, use_name=True),
        )
        metrics_dict[task_name]["name_shape_triplet_f1"].update(
            reference=ref, prediction=pred
        )

    elif task_name == "FlowChart2JSON":
        pred, ref = process_pred_and_reference(
            get_pred(result),
            get_reference(result),
            partial(extract_triplet_from_json, use_shape=True, use_name=True),
        )
        metrics_dict[task_name]["name_shape_triplet_f1"].update(
            reference=ref, prediction=pred
        )

    # elif task_name in ['chart_caption', 'chart2summary', 'gui_user_intent_qa', 'webui_qa', 'gui_summary']:
    elif task_name in [
        "Chart2Caption",
        "GUI2Summary",
        "GUI2UserIntent",
        "GUIVQA",
        "GUI2Summary",
    ]:
        pred, ref = get_pred(result), get_reference(result)
        metrics_dict[task_name]["rouge"].update(reference=ref, prediction=pred)

    elif task_name == "screenshot2html":
        pred, ref = get_pred(result), get_reference(result)
        metrics_dict[task_name]["similarity_scores"].update(
            reference=ref, prediction=pred
        )

    elif task_name == "GUI2BBox":
        pred, ref = get_pred(result), get_reference(result, keep_original=True)
        metrics_dict[task_name]["bbox_iou"].update(reference=ref, prediction=pred)


class BigDocsBench(ImageBaseDataset):
    TYPE = "VQA"

    URL_PREFIX = "https://huggingface.co/datasets/BigDocs/test_tsv_dataset/resolve/main"

    DATASET_URL = {
        "BigDocsBench": f"{URL_PREFIX}/bigdocs.xlsx",
        "BigDocsBench_Chart2Caption": f"{URL_PREFIX}/Chart2Caption.tsv",
        "BigDocsBench_Chart2Markdown": f"{URL_PREFIX}/Chart2Markdown.tsv",
        "BigDocsBench_GUIVQA": f"{URL_PREFIX}/GUIVQA.tsv",
        "BigDocsBench_GUI2BBox": f"{URL_PREFIX}/GUI2BBox.tsv",
        "BigDocsBench_GUI2Summary": f"{URL_PREFIX}/GUI2Summary.tsv",
        "BigDocsBench_GUI2UserIntent": f"{URL_PREFIX}/GUI2UserIntent.tsv",
        "BigDocsBench_FlowChart2Graphviz": f"{URL_PREFIX}/FlowChart2Graphviz.tsv",
        "BigDocsBench_FlowChart2JSON": f"{URL_PREFIX}/FlowChart2JSON.tsv",
        "BigDocsBench_Image2SVG": f"{URL_PREFIX}/Image2SVG.tsv",
        "BigDocsBench_Text2SVG": f"{URL_PREFIX}/Text2SVG.tsv",
        "BigDocsBench_Screenshot2HTML": f"{URL_PREFIX}/Screenshot2HTML.tsv",
        "BigDocsBench_Table2Latex": f"{URL_PREFIX}/Table2Latex.tsv",
    }

    DATASET_MD5 = {
        "BigDocsBench": "f7e4b7312b6808805226051f884f5bc5",
        "BigDocsBench_Chart2Caption": "cc9a5bac243bd8fdcfe0ec95e0344d0d",
        "BigDocsBench_Chart2Markdown": "771ed76224a343b9ca783f912ef1efae",
        "BigDocsBench_GUIVQA": "ff1cc99dcb885d80eb40387d4b0842a5",
        "BigDocsBench_GUI2BBox": "491fb59a6764ea82604bbac16f0016eb",
        "BigDocsBench_GUI2Summary": "a2c4605a161d37bba8910f95bc5b6304",
        "BigDocsBench_GUI2UserIntent": "25861c7c124a8140f4e0d31e0d1bf57f",
        "BigDocsBench_FlowChart2Graphviz": "b3920766726573748d5d56babd35e4b7",
        "BigDocsBench_FlowChart2JSON": "29003c0690aebb1fc77752ec8d79367f",
        "BigDocsBench_Image2SVG": "ee7b29c8ecce1ecab896aa80407879a4",
        "BigDocsBench_Text2SVG": "463f21d2161367a22f309f674b1044b6",
        "BigDocsBench_Screenshot2HTML": "5c42104f8a42bcd43d125b7651b8531e",
        "BigDocsBench_Table2Latex": "a381ee311822f8c6a68459e36f9ff04f",
    }

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        logger = get_logger("Evaluation")
        data = load(eval_file)

        metrics_dict = {
            "Table2Latex": {"texbleu": TexBLEUMetric()},
            "Chart2Markdown": {"rmsf1": RMSF1Metric()},
            "Text2SVG": {"dino2score": DINO2ScoreMetric()},
            "Image2SVG": {"dino2score": DINO2ScoreMetric()},
            "FlowChart2Graphviz": {"name_shape_triplet_f1": TripletF1Metric()},
            "FlowChart2JSON": {"name_shape_triplet_f1": TripletF1Metric()},
            "Chart2Caption": {"rouge": RougeMetric()},
            "GUI2Summary": {"rouge": RougeMetric()},
            "GUI2UserIntent": {"rouge": RougeMetric()},
            "GUIVQA": {"rouge": RougeMetric()},
            "Screenshot2HTML": {"similarity_scores": HTMLSimilarityMetric()},
            "GUI2BBox": {"bbox_iou": BBoxIOUMetric()},
        }

        metrics_dict = {task: metrics_dict[task] for task in ALL_TASKS}

        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]

        # get the set of 'category' in the dataset
        existing_categories = set([line["category"] for line in lines])
        assert all([category in ALL_TASKS for category in existing_categories])
        metrics_dict = {
            category: metrics_dict[category]
            for category in metrics_dict.keys()
            if category in existing_categories
        }

        for instance_id, instance in tqdm(enumerate(lines)):
            if instance["category"] in ALL_TASKS:
                evaluate_per_task(instance, metrics_dict)

        results = {
            task: {
                metric: metrics_dict[task][metric].compute()
                for metric in metrics_dict[task].keys()
            }
            for task in metrics_dict.keys()
        }
        results = iterative_tensor_to_float(results)

        score_pth_json = eval_file.replace(".xlsx", f"_score.json")
        dump(results, score_pth_json)
        logger.info(
            f"BigDocsBench successfully finished evaluating {eval_file}, results saved in {score_pth_json}"
        )
        logger.info("Score: ")
        for key, value in results.items():
            logger.info("{}:{}".format(key, value))

        results_pd_dict = {
            task: list(results[task].values())[0] for task in results.keys()
        }
        results_pd = pd.DataFrame(results_pd_dict)
        return results_pd
