import os
import re
import time
from typing import Dict

import openai
import torch
from torchmetrics import Metric

from vlmeval.dataset.utils.bigdocs.metrics_utils import bootstrap_std

# ==================== Azure OpenAI Configuration ====================

# Replace these with your actual Azure OpenAI credentials
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = "https://stardoc.openai.azure.com/"  # Your Azure OpenAI endpoint
api_version = "2024-04-01-preview"  # The API version you are using
deployment_name = "gpt-4o-mini"  # Your deployment name

# Configure the OpenAI package to use Azure OpenAI
openai.api_type = "azure"
openai.api_base = azure_endpoint
openai.api_version = api_version
openai.api_key = api_key


# =====================================================================


class GPT4SimilarityMetric(Metric):
    """
    TorchMetrics class to evaluate the similarity between predicted text and ground-truth text
    using OpenAI's GPT-4 model.
    """

    def __init__(self,
                 task: str,
                 model: str = deployment_name,
                 max_retries: int = 3,
                 sleep_time: float = 1.0):
        super().__init__()

        supported_tasks = ['chart_caption', 'chart2summary', 'gui_user_intent_qa', 'webui_qa', 'gui_summary']
        if task not in supported_tasks:
            raise ValueError(f"Unsupported task '{task}'. Supported tasks are: {supported_tasks}")

        self.task = task
        self.model = model
        self.max_retries = max_retries
        self.sleep_time = sleep_time

        self.task_to_description = {
            'chart_caption': "chart captioning",
            'chart2summary': "chart summarization",
            'gui_user_intent_qa': "GUI user intent question answering",
            'webui_qa': "Web UI question answering",
            'gui_summary': "GUI summarization"
        }

        # Define states
        self.add_state("total_similarity", default=list(), dist_reduce_fx="cat")

    def _get_similarity_score(self, prompt: str) -> float:
        for attempt in range(self.max_retries):

            try:
                response = openai.ChatCompletion.create(
                    engine=deployment_name,
                    messages=[
                        {"role": "system",
                         "content": f"You are an evaluator for {self.task_to_description[self.task]}."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0  # Ensures deterministic output
                )

                # Extract the content from the response
                answer = response.choices[0].message.content.strip()

            except openai.error.InvalidRequestError:
                return 1.0
            except Exception:
                continue

            # Extract the first number in the response as the score

            match = re.search(r'\b([0-5](?:\.\d+)?)\b', answer)
            if match:
                score = float(match.group(1))
                if 1 <= score <= 5:
                    return score
            #     else:
            #         print(f"Received out-of-range score: {score}. Retrying...")
            # else:
            #     print(f"No valid score found in response: '{answer}'. Retrying...")

            # Wait before retrying
            time.sleep(self.sleep_time)

        # If all retries fail, assign a default score (e.g., 1) or handle as needed
        return 1.0

    def update(self, reference: str, prediction: str) -> None:
        if not isinstance(reference, str) or not isinstance(prediction, str):
            raise ValueError("Both reference and prediction must be strings.")
        if len(reference) > 0 and len(prediction) > 0:
            prompt = self._build_prompt(reference, prediction)
            score = self._get_similarity_score(prompt)
            self.total_similarity.append(score)

    def compute(self) -> Dict[str, torch.Tensor]:
        if len(self.total_similarity) == 0:
            return {"gpt4score_mean": torch.tensor(0.0), "gpt4score_std": torch.tensor(0.0)}
        similarity = torch.tensor(self.total_similarity)
        similarity = (similarity - 1) / 4  # Normalize to [0, 1]
        std, lb, ub = bootstrap_std(similarity, n_bootstrap=1000, ci=0.95)
        return {"gpt4score_mean": similarity.mean(), "gpt4score_std": std}

    def _build_prompt(self, reference: str, prediction: str) -> str:
        """
        Constructs the prompt to be sent to GPT-4 based on the task.

        Args:
            reference (str): The ground-truth text.
            prediction (str): The machine-generated text.

        Returns:
            str: The formatted prompt.
        """
        prompt_templates = {
            'chart_caption': """You are given a caption generated by a model based on a data chart. Your task is to evaluate the model-generated caption against a ground truth caption (reference). Focus on the following criteria:
1. Caption Accuracy: Compare the generated caption with the reference caption. How well does it capture the essential insights from the chart, such as key trends, data points, relationships, or other important chart elements? Identify if there are any omissions, misinterpretations, or incorrect details in the generated caption compared to the reference.
2. Clarity and Conciseness: Assess the clarity and conciseness of the generated caption. Is the information conveyed in a clear and easy-to-understand manner without irrelevant or redundant details?

Use the following scoring rubric:

5: The generated caption perfectly matches the reference caption, accurately describing all key insights and features of the chart with clear and concise language. No irrelevant or incorrect information is present.
4: The generated caption is mostly accurate with minor differences from the reference caption, such as slight omissions or slight misinterpretations that do not significantly affect the understanding of the chart. The caption remains clear and concise.
3: The generated caption contains several inaccuracies or omissions compared to the reference caption, but the main message of the chart is still conveyed. There may be some issues with clarity or verbosity.
2: The generated caption contains significant errors or omissions that lead to misunderstanding key aspects of the chart when compared to the reference. The caption may be unclear or contain irrelevant details.
1: The generated caption fails to capture the key elements of the chart, with major inaccuracies or irrelevant, unclear information compared to the reference caption.

The output should be a JSON file with two fields: "score" and "feedback", explaining the reason behind your evaluation, specifically comparing the generated caption to the reference caption.
Model output: {pred}
Reference Caption: {gt}
Output:""",
            'chart2summary': """You are given a summary generated by a model based on an input data chart. Your task is to evaluate the summary output based on the following criteria:
1. Summary Accuracy: How well does the generated summary capture the key insights and trends from the reference chart (e.g., main data patterns, peaks, troughs, correlations, etc.)?
2. Clarity and Precision: Evaluate the clarity, conciseness, and correctness of the summary. Does it clearly convey the main points without unnecessary information or errors?
You should use the following scoring rubric:

5: Perfect summary, capturing all key insights and trends from the chart with precise and clear language. No irrelevant or incorrect information.
4: Mostly accurate, with minor omissions or slight misinterpretations that do not significantly affect the overall understanding of the chart. Mostly clear and correct.
3: Several noticeable omissions or inaccuracies, but the general trend and message of the chart are still conveyed. Could be more concise or clear.
2: Significant errors or omissions that lead to a misunderstanding of key data points or trends in the chart. The summary lacks clarity or includes irrelevant details.
1: Very poor summary that fails to capture the main points and trends of the chart, with major inaccuracies or unclear writing.

The output should be a JSON file with two fields: "score" and "feedback", explaining the reason behind your evaluation.
Model output: {pred}
Reference Chart Summary: {gt}
Output:""",
            'gui_user_intent_qa': """You are given a model-generated answer and a ground truth (reference) regarding a user's intent based on a GUI screenshot. Your task is to evaluate how well the model-generated answer captures the user's intent when compared to the provided reference answer. Focus on the following criteria:

1. Intent Alignment: How closely does the model’s answer align with the ground truth in terms of identifying the correct user intent (e.g., the action, function, or goal the user is trying to achieve based on the GUI)?
2. Content Clarity and Relevance: Evaluate whether the model's answer is clear, concise, and free of irrelevant or misleading information. Does the answer provide a coherent explanation of the user's intent without introducing unnecessary details?

Scoring Rubric:

5: The model’s answer perfectly matches the ground truth, with a precise and clear explanation of the user’s intent. No irrelevant or incorrect details are present.
4: The model’s answer is mostly accurate, with minor misinterpretations or omissions that do not significantly alter the understanding of the user’s intent. The explanation is still clear and relevant.
3: The model’s answer contains noticeable inaccuracies or missing elements, but the general intent is captured. The answer may lack some clarity or conciseness.
2: The model’s answer includes significant errors or misunderstandings of the user's intent. The answer may include irrelevant details or lack clarity, causing confusion.
1: The model’s answer fails to capture the user’s intent, with major inaccuracies or irrelevant information.

The output should be a JSON file with two fields: "score" and "feedback", explaining the reason behind your evaluation.

Model output: {pred}
Reference User Intent: {gt}
Output:""",
            'webui_qa': """You are given an answer generated by a model based on a question about a Web UI, along with the ground truth reference answer. Your task is to evaluate the model's answer to the question against the reference answer using the following criteria:

1. Answer Accuracy: Compare the generated answer with the ground truth. Does the generated answer accurately reflect the information found in the ground truth? Focus on whether the key elements, actions, or features relevant to the question are correctly identified and described.
2. Clarity and Relevance: Assess the clarity, conciseness, and relevance of the generated answer. Is the information clearly presented, directly relevant to the question, and devoid of unnecessary or incorrect details?

You should use the following scoring rubric to assign a score:

5: Perfect match with the reference answer. The generated answer is accurate, directly addresses the question, and is clear and concise with no irrelevant or incorrect information.
4: Mostly matches the reference answer with minor misinterpretations or omissions that do not significantly affect the overall meaning. The answer is mostly clear and relevant.
3: Some noticeable differences or omissions from the reference answer, but the main idea is captured. The answer may lack clarity or conciseness.
2: Significant differences or misunderstandings compared to the reference answer. The generated answer may contain irrelevant or unclear details.
1: Fails to match the reference answer with major inaccuracies, irrelevance, or lack of clarity.

The output should be a JSON file with two fields: "score" and "feedback", explaining the reason behind your evaluation.

Model output: {pred}
Reference Answer: {gt}
Output:""",
            'gui_summary': """You are given a summary generated by a model and a ground-truth (reference) summary based on a screenshot of a GUI. Your task is to evaluate how well the generated summary matches the ground-truth summary by considering the following criteria:

1. Summary Accuracy: Assess how well the generated summary aligns with the ground-truth summary. Focus on whether the model output captures the same key elements and functionality described in the ground truth, including features, layout, and interactions.
2. Clarity and Conciseness: Evaluate the clarity and brevity of the generated summary. Determine if the model’s output conveys the essential aspects without unnecessary details or confusing information.
3. Comparative Fidelity: Compare the model output to the reference summary directly. Does the generated summary omit critical information, or does it introduce new content that is not mentioned in the ground truth? How closely does it adhere to the reference summary?

You should use the following scoring rubric:

5: Perfect alignment with the ground-truth summary. Captures all key elements and functionality in a clear and concise manner, with no irrelevant or incorrect information.
4: Mostly accurate compared to the ground-truth, with minor omissions or slight misinterpretations that do not significantly affect the overall alignment. Mostly clear and concise.
3: Several noticeable omissions or discrepancies with the ground-truth summary, but the general purpose and structure are conveyed. Some issues with clarity or unnecessary details.
2: Significant errors or omissions when compared to the ground-truth, leading to misunderstanding of the main elements or functionality. The summary may lack clarity or include irrelevant information.
1: Poor alignment with the ground-truth. Major inaccuracies, omissions, or irrelevant information that fail to convey the key elements of the GUI.

The output should be a JSON file with two fields: "score" and "feedback", explaining the reason behind your evaluation.

Model output: {pred}
Reference Summary: {gt}
Output:"""
        }

        if self.task not in prompt_templates:
            raise ValueError(f"No prompt template defined for task '{self.task}'.")

        prompt = prompt_templates[self.task]
        prompt = prompt.replace("{gt}", reference.strip())
        prompt = prompt.replace("{pred}", prediction.strip())
        return prompt
