import itertools
import json
import os
import re
from functools import partial
from typing import Optional, Tuple
from warnings import filterwarnings

import torch
from fire import Fire
from tqdm import tqdm

# from stardoc.validation.metrics.bigdocs import GPT4SimilarityMetric
from vlmeval.dataset.utils.bigdocs.metrics_utils import json_to_csv_split_headers

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
    return example.get('model_output', '')


def get_reference(example, keep_original=False):
    reference = example.get('gpt_answer', dict()).get('content', '')
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


def evaluate_per_task(results, task_name, total_instances=500):
    assert task_name in ALL_TASKS, f"Invalid task name: {task_name}"

    assert len(results) == total_instances, f"Expected {total_instances} instances, got {len(results)}"

    metrics_dict = dict()
    if task_name == 'table2latex':
        from stardoc.validation.metrics.bigdocs.metrics_utils import preprocess_latex_table
        from stardoc.validation.metrics.bigdocs import TexBLEUMetric, RougeMetric
        texbleu_metric = TexBLEUMetric()
        rouge_metric = RougeMetric()
        metrics_dict = {'texbleu': texbleu_metric, 'rouge': rouge_metric}
        for result in tqdm(results, desc="Evaluating table2latex task", leave=False):
            pred, ref = process_pred_and_reference(get_pred(result), get_reference(result), preprocess_latex_table)
            texbleu_metric.update(reference=ref, prediction=pred)
            rouge_metric.update(target=ref, preds=pred)

    elif task_name == 'chart2md':
        from stardoc.validation.metrics.bigdocs.metrics_utils import preprocess_markdown
        from stardoc.validation.metrics.bigdocs import RMSF1Metric, SacreBLEUMetric, RougeMetric
        rmsf1_metric = RMSF1Metric()
        sacrebleu_metric = SacreBLEUMetric()
        rouge_metric = RougeMetric()
        metrics_dict = {'rmsf1': rmsf1_metric, 'sacrebleu': sacrebleu_metric, 'rouge': rouge_metric}
        for result in tqdm(results, desc="Evaluating chart2md task", leave=False):
            pred, ref = process_pred_and_reference(get_pred(result), get_reference(result), preprocess_markdown)
            rmsf1_metric.update(reference=ref, prediction=pred)
            sacrebleu_metric.update(target=ref, preds=pred)
            rouge_metric.update(target=ref, preds=pred)

    elif task_name in ['text2svg', 'image2svg']:
        from stardoc.validation.metrics.bigdocs.metrics_utils import validate_plot_svg
        from stardoc.validation.metrics.bigdocs import DINO2ScoreMetric
        dino2score_metric = DINO2ScoreMetric()
        metrics_dict = {'dino2score': dino2score_metric}
        for result in tqdm(results, desc=f"Evaluating {task_name} task", leave=False):
            pred, ref = process_pred_and_reference(get_pred(result), get_reference(result), validate_plot_svg)
            dino2score_metric.update(reference=ref, prediction=pred)

    elif task_name == 'flow2graphviz':
        from stardoc.validation.metrics.bigdocs.metrics_utils import extract_triplet_from_graphviz_pydot
        from stardoc.validation.metrics.bigdocs import TripletF1Metric
        triplet_f1_name_only_metric = TripletF1Metric()
        triplet_f1_name_shape_metric = TripletF1Metric()
        metrics_dict = {'name_triplet_f1': triplet_f1_name_only_metric,
                        'name_shape_triplet_f1': triplet_f1_name_shape_metric}
        for result in tqdm(results, desc="Evaluating flow2graphviz task", leave=False):
            pred, ref = process_pred_and_reference(get_pred(result), get_reference(result),
                                                   partial(extract_triplet_from_graphviz_pydot,
                                                           use_shape=False,
                                                           use_name=True))
            triplet_f1_name_only_metric.update(reference=ref, prediction=pred)
            pred, ref = process_pred_and_reference(get_pred(result), get_reference(result),
                                                   partial(extract_triplet_from_graphviz_pydot,
                                                           use_shape=True,
                                                           use_name=True))
            triplet_f1_name_shape_metric.update(reference=ref, prediction=pred)

    elif task_name == 'flow2json':
        from stardoc.validation.metrics.bigdocs.metrics_utils import extract_triplet_from_json
        from stardoc.validation.metrics.bigdocs import TripletF1Metric
        triplet_f1_name_only_metric = TripletF1Metric()
        triplet_f1_name_shape_metric = TripletF1Metric()
        metrics_dict = {'name_triplet_f1': triplet_f1_name_only_metric,
                        'name_shape_triplet_f1': triplet_f1_name_shape_metric}
        for result in tqdm(results, desc="Evaluating flow2json task", leave=False):
            pred, ref = process_pred_and_reference(get_pred(result), get_reference(result),
                                                   partial(extract_triplet_from_json,
                                                           use_shape=False,
                                                           use_name=True))
            triplet_f1_name_only_metric.update(reference=ref, prediction=pred)
            pred, ref = process_pred_and_reference(get_pred(result), get_reference(result),
                                                   partial(extract_triplet_from_json,
                                                           use_shape=True,
                                                           use_name=True))
            triplet_f1_name_shape_metric.update(reference=ref, prediction=pred)

    elif task_name in ['chart_caption', 'chart2summary', 'gui_user_intent_qa', 'webui_qa', 'gui_summary']:
        from stardoc.validation.metrics.bigdocs import SacreBLEUMetric, RougeMetric, BERTScoreMetric
        sacrebleu_metric = SacreBLEUMetric()
        rouge_metric = RougeMetric()
        bert_score_metric = BERTScoreMetric(
            device=(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')),
            model_name_or_path="roberta-large")
        # gpt4_metric = GPT4SimilarityMetric(task=task_name, model="gpt-4o-mini-2024-07-18", )
        metrics_dict = {'sacrebleu': sacrebleu_metric,
                        'rouge': rouge_metric,
                        # 'gpt4': gpt4_metric,
                        'bert_score': bert_score_metric}
        for result in tqdm(results, desc=f"Evaluating {task_name} task", leave=False):
            pred, ref = get_pred(result), get_reference(result)
            sacrebleu_metric.update(target=ref, preds=pred)
            rouge_metric.update(target=ref, preds=pred)
            bert_score_metric.update(target=ref, preds=pred)
            # gpt4_metric.update(reference=ref, prediction=pred)

    elif task_name == 'screenshot2html':
        from stardoc.validation.metrics.bigdocs import HTMLSimilarityMetric
        html_metric = HTMLSimilarityMetric()
        metrics_dict = {'similarity_scores': html_metric}
        for result in tqdm(results, desc="Evaluating screenshot2html task", leave=False):
            pred, ref = get_pred(result), get_reference(result)
            html_metric.update(reference=ref, prediction=pred)

    elif task_name == 'gui2bbox':
        from stardoc.validation.metrics.bigdocs import BBoxIOUMetric
        iou_metric = BBoxIOUMetric()
        metrics_dict = {'bbox_iou': iou_metric}
        for result in tqdm(results, desc="Evaluating gui2bbox task", leave=False):
            pred, ref = get_pred(result), get_reference(result, keep_original=True)
            iou_metric.update(reference=ref, prediction=pred)

    results = {metric: metrics_dict[metric].compute() for metric in metrics_dict.keys()}
    results = iterative_tensor_to_float(results)
    # print(f" --- Results for {task_name} task:")
    # print(results)
    return results


def main(result_path: str = 'stardocintel/metrics/bigdocs/tmp_outputs.json',
         old_task_names: bool = False,
         skip_tasks: Optional[Tuple[str]] = None,
         run_tasks: Optional[Tuple[str]] = None):
    """
    Evaluate the results of the BigDoc tasks.
    :param result_path: Path to the json file containing the results.
    :param old_task_names: Whether the results are from the old task names.
    :param skip_tasks: Tuple of task names to skip.
    :param run_tasks: Tuple of task names to run. If not provided, all tasks will be run.
    :return: Dictionary containing the evaluation results for each task.
    """
    # read the json file
    assert not (skip_tasks and run_tasks), "Only one of skip_tasks or run_tasks can be provided."

    if run_tasks is not None:
        print(f"Run tasks set. Only run the following tasks: {run_tasks}")
    if skip_tasks is not None:
        print(f"Skip tasks set. Not evaluating the following tasks: {skip_tasks}")
    print(f"Reading results from {result_path}")
    with open(result_path, 'r') as f:
        results = json.load(f)

    if old_task_names:
        results_dict = dict()
        # results_dict['table2latex'] = results.get('table2latex', dict()).get('pdf2latex', None)
        # results_dict['chart2md'] = results.get('chart_caption', dict()).get('NovelCharts', None)
        # results_dict['text2svg'] = results.get('text2svg', dict()).get('SVGDataset', None)
        # results_dict['image2svg'] = results.get('image2svg', dict()).get('SVGDataset', None)
        # results_dict['flow2graphviz'] = results.get('flow2graphviz', dict()).get('BigDocImage2Flow', None)
        # results_dict['flow2json'] = results.get('flow2json', dict()).get('BigDocImage2Flow', None)
        # results_dict['chart_caption'] = results.get('chart_caption', dict()).get('NovelCharts', None)
        # results_dict['chart2summary'] = results.get('chart2summary', dict()).get('BigDocChart2Summary', None)
        # results_dict['screenshot2html'] = results.get('screenshot2html', dict()).get('Screenshot2HTML', None)
        # results_dict['gui_user_intent_qa'] = results.get('gui_user_intent_qa', dict()).get('gui_user_intent', None)
        # results_dict['webui_qa'] = results.get('webui_qa', dict()).get('webui_qa', None)
        # results_dict['gui_summary'] = results.get('gui_summary', dict()).get('gui_summ', None)
        results_dict['table2latex'] = list(
            itertools.chain.from_iterable([v for k, v in results.items() if re.fullmatch(r'\d{4}\.\d{5}v\d', k)]))
        results_dict['chart2md'] = [l for l in results.get('NovelCharts', [dict()]) if
                                    l.get('task_name', '') == 'chart2md']
        results_dict['text2svg'] = [l for l in results.get('SVGDataset', [dict()]) if
                                    l.get('task_name', '') == 'text2svg']
        results_dict['image2svg'] = [l for l in results.get('SVGDataset', [dict()]) if
                                     l.get('task_name', '') == 'image2svg']
        results_dict['flow2graphviz'] = [l for l in results.get('BigDocImage2Flow', [dict()]) if
                                         l.get('task_name', '') == 'flow2graphviz']
        results_dict['flow2json'] = [l for l in results.get('BigDocImage2Flow', [dict()]) if
                                     l.get('task_name', '') == 'flow2json']
        results_dict['chart_caption'] = [l for l in results.get('NovelCharts', [dict()]) if
                                         l.get('task_name', '') == 'chart_caption']
        results_dict['chart2summary'] = [l for l in results.get('BigDocChart2Summary', [dict()]) if
                                         l.get('task_name', '') == 'chart2summary']
        results_dict['screenshot2html'] = [l for l in results.get('Screenshot2HTML', [dict()]) if
                                           l.get('task_name', '') == 'screenshot2html']
        results_dict['gui_user_intent_qa'] = [l for l in results.get('gui_user_intent', [dict()]) if
                                              l.get('task_name', '') == 'gui_user_intent_qa']
        results_dict['webui_qa'] = [l for l in results.get('webui_qa', [dict()]) if
                                    l.get('task_name', '') == 'webui_qa']
        results_dict['gui_summary'] = [l for l in results.get('gui_summ', [dict()]) if
                                       l.get('task_name', '') == 'gui_summary']
        results_dict['gui2bbox'] = [l for l in results.get('GUI2BBox', [dict()]) if
                                    l.get('task_name', '') == 'GUI2BBox']
        for k in results_dict:
            if len(results_dict[k]) == 0:
                results_dict[k] = None
    else:
        results_dict = results

    results_dict = {k: v for k, v in results_dict.items() if v is not None}
    eval_results = {}
    for task_name in tqdm(ALL_TASKS, desc="Evaluating tasks"):
        if task_name not in results_dict:
            print(f"Skipping {task_name} task as it is not present in the results.")
            continue
        elif skip_tasks is not None and task_name in skip_tasks:
            print(f"Skipping {task_name} task as it is in the skip_tasks list.")
            continue
        elif run_tasks is not None and task_name not in run_tasks:
            print(f"Skipping {task_name} task as it is not in the run_tasks list.")
            continue
        prediction_results = results_dict[task_name]
        metrics = evaluate_per_task(prediction_results, task_name, len(prediction_results))
        eval_results[task_name] = metrics

    if run_tasks is not None or skip_tasks is not None:
        result_folder = 'part_results'
    else:
        result_folder = 'results'

    os.makedirs(os.path.join(os.path.dirname(result_path), result_folder), exist_ok=True)
    input_filename = os.path.basename(result_path)

    result_json_path = os.path.join(os.path.dirname(result_path), result_folder,
                                    input_filename.replace('.json', '_evaluate_result.json'))
    with open(result_json_path, 'w') as f:
        json.dump(eval_results, f, indent=4)
    print(f"Results json saved in {result_json_path}")

    csv_path = os.path.join(os.path.dirname(result_path), result_folder,
                            input_filename.replace('.json', '_evaluate_result.csv'))
    json_to_csv_split_headers(eval_results, csv_path)
    print(f"Results csv saved in {csv_path}")


if __name__ == '__main__':
    filterwarnings("ignore")

    results = Fire(main)
