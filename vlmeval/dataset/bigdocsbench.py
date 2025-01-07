from functools import partial

import pandas as pd
import torch
from tqdm import tqdm

from vlmeval.dataset.utils.bigdocs import (BBoxIOUMetric,
                                           DINO2ScoreMetric,
                                           HTMLSimilarityMetric,
                                           RMSF1Metric,
                                           RougeMetric,
                                           TexBLEUMetric,
                                           TripletF1Metric)
from vlmeval.dataset.utils.bigdocs.metrics_utils import (extract_triplet_from_graphviz_pydot,
                                                         extract_triplet_from_json,
                                                         preprocess_latex_table,
                                                         preprocess_markdown,
                                                         validate_plot_svg)
from .image_base import ImageBaseDataset
from ..smp import get_logger, load, dump

ALL_TASKS = ['table2latex',
             'chart2md',
             'text2svg',
             'image2svg',
             'flow2graphviz',
             'flow2json',
             'chart_caption',
             'chart2summary',
             'screenshot2html',
             'gui_user_intent_qa',
             'webui_qa',
             'gui_summary',
             'gui2bbox',
             ]


def get_pred(example):
    return example.get('prediction', '')


def get_reference(example, keep_original=False):
    reference = example.get('answer', '')
    if isinstance(reference, list):
        reference = reference[0]
    if isinstance(reference, dict):
        reference = reference.get('text', '')

    if not keep_original:
        if reference.startswith('[') and reference.endswith(']'):
            reference = reference[1:-1]
        if reference.startswith('"') and reference.endswith('"'):
            reference = reference[1:-1]
        if reference.startswith("'") and reference.endswith("'"):
            reference = reference[1:-1]
        reference = reference.replace(r'\n', '\n').replace(r'\t', '\t')
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
    task_name = result.get('category', '')
    assert task_name in ALL_TASKS, f"Invalid task name: {task_name}"

    if task_name == 'table2latex':
        pred, ref = process_pred_and_reference(get_pred(result), get_reference(result), preprocess_latex_table)
        metrics_dict[task_name]['texbleu'].update(reference=ref, prediction=pred)

    elif task_name == 'chart2md':
        pred, ref = process_pred_and_reference(get_pred(result), get_reference(result), preprocess_markdown)
        metrics_dict[task_name]['rmsf1'].update(reference=ref, prediction=pred)

    elif task_name in ['text2svg', 'image2svg']:
        pred, ref = process_pred_and_reference(get_pred(result), get_reference(result), validate_plot_svg)
        metrics_dict[task_name]['dino2score'].update(reference=ref, prediction=pred)

    elif task_name == 'flow2graphviz':
        pred, ref = process_pred_and_reference(get_pred(result), get_reference(result),
                                               partial(extract_triplet_from_graphviz_pydot,
                                                       use_shape=True,
                                                       use_name=True))
        metrics_dict[task_name]['name_shape_triplet_f1'].update(reference=ref, prediction=pred)

    elif task_name == 'flow2json':
        pred, ref = process_pred_and_reference(get_pred(result), get_reference(result),
                                               partial(extract_triplet_from_json,
                                                       use_shape=True,
                                                       use_name=True))
        metrics_dict[task_name]['name_shape_triplet_f1'].update(reference=ref, prediction=pred)

    elif task_name in ['chart_caption', 'chart2summary', 'gui_user_intent_qa', 'webui_qa', 'gui_summary']:
        pred, ref = get_pred(result), get_reference(result)
        metrics_dict[task_name]['rouge'].update(reference=ref, prediction=pred)

    elif task_name == 'screenshot2html':
        pred, ref = get_pred(result), get_reference(result)
        metrics_dict[task_name]['similarity_scores'].update(reference=ref, prediction=pred)

    elif task_name == 'gui2bbox':
        pred, ref = get_pred(result), get_reference(result, keep_original=True)
        metrics_dict[task_name]['bbox_iou'].update(reference=ref, prediction=pred)


class BigDocsBench(ImageBaseDataset):
    TYPE = 'VQA'

    DATASET_URL = {
        'BigDocsBench': f'https://litter.catbox.moe/a64k6e.tsv',
        'BigDocsBench_Flow': f'https://litter.catbox.moe/7z7k6e.tsv',
    }

    DATASET_MD5 = {
        'BigDocsBench': 'ff53e6e8913e71a9f5f79c4e55d260f8',
        'BigDocsBench_Flow': '9a1b1b0b6e2d1d2f2c2d2d2f'
    }

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        logger = get_logger('Evaluation')
        data = load(eval_file)

        metrics_dict = {
            'table2latex': {'texbleu': TexBLEUMetric()},
            'chart2md': {'rmsf1': RMSF1Metric()},
            'text2svg': {'dino2score': DINO2ScoreMetric()},
            'image2svg': {'dino2score': DINO2ScoreMetric()},
            'flow2graphviz': {'name_shape_triplet_f1': TripletF1Metric()},
            'flow2json': {'name_shape_triplet_f1': TripletF1Metric()},
            'chart_caption': {'rouge': RougeMetric()},
            'chart2summary': {'rouge': RougeMetric()},
            'gui_user_intent_qa': {'rouge': RougeMetric()},
            'webui_qa': {'rouge': RougeMetric()},
            'gui_summary': {'rouge': RougeMetric()},
            'screenshot2html': {'similarity_scores': HTMLSimilarityMetric()},
            'gui2bbox': {'bbox_iou': BBoxIOUMetric()},
        }

        metrics_dict = {
            task: metrics_dict[task] for task in ALL_TASKS
        }

        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]

        # get the set of 'category' in the dataset
        existing_categories = set([line['category'] for line in lines])
        assert all([category in ALL_TASKS for category in existing_categories])
        metrics_dict = {category: metrics_dict[category] for category in metrics_dict.keys() if
                        category in existing_categories}

        for instance_id, instance in tqdm(enumerate(lines)):
            if instance['category'] in ALL_TASKS:
                evaluate_per_task(instance, metrics_dict)

        results = {task: {metric: metrics_dict[task][metric].compute() for metric in metrics_dict[task].keys()}
                   for task in metrics_dict.keys()}
        results = iterative_tensor_to_float(results)

        score_pth_json = eval_file.replace(
            '.xlsx', f'_score.json'
        )
        dump(results, score_pth_json)
        logger.info(
            f'BigDocsBench successfully finished evaluating {eval_file}, results saved in {score_pth_json}'
        )
        logger.info('Score: ')
        for key, value in results.items():
            logger.info('{}:{}'.format(key, value))

        results_pd_dict = {task: list(results[task].values())[0] for task in results.keys()}
        results_pd = pd.DataFrame(results_pd_dict)
        return results_pd
