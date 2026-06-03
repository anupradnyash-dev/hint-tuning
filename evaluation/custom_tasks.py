"""
Custom lighteval task definitions for Hint Tuning evaluation.

Requires:
  pip install lighteval[vllm] inspect-ai
"""

import os

import numpy as np
from transformers import AutoTokenizer
from lighteval.tasks.tasks.aime import MATH_PROMPT_TEMPLATE, record_to_sample
from lighteval.tasks.tasks.math_500 import MATH_QUERY_TEMPLATE, record_to_sample as math500_record_to_sample
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.requests import Doc, SamplingMethod
from inspect_ai.solver import prompt_template, generate
from inspect_ai.scorer import model_graded_fact
from lighteval.metrics.metrics import math_scorer
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.models.model_output import ModelResponse


def get_tokenizer():
    """Load tokenizer from EVAL_MODEL_PATH environment variable."""
    model_path = os.environ.get("EVAL_MODEL_PATH")
    if not model_path:
        raise ValueError("EVAL_MODEL_PATH environment variable is not set")
    print(f"Loading tokenizer from: {model_path}")
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


_TOKENIZER = None


class TokenLengthMetric(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        global _TOKENIZER
        if _TOKENIZER is None:
            _TOKENIZER = get_tokenizer()
        if not model_response.text:
            return 0.0
        counts = [len(_TOKENIZER.encode(text, add_special_tokens=False)) for text in model_response.text]
        return float(np.mean(counts))


output_token_metric = SampleLevelMetric(
    metric_name="output_tokens",
    sample_level_fn=TokenLengthMetric(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=False,
)


# ─── Custom prompt functions ──────────────────────────────────────────────────

def custom_aime_prompt(line, task_name: str = None):
    problem = (
        f"{line['problem']}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    return Doc(task_name=task_name, query=problem, choices=[line["answer"]], gold_index=0)


def custom_math_500_prompt(line, task_name: str = None):
    problem = (
        f"{line['problem']}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    return Doc(task_name=task_name, query=problem, choices=[f"ANSWER: {line['solution']}"], gold_index=0)


# ─── Task configs ─────────────────────────────────────────────────────────────

aime24_local = LightevalTaskConfig(
    name="aime24:local",
    prompt_function=custom_aime_prompt,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
    hf_repo="AI-MO/aimo-validation-aime",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[
        Metrics.pass_at_k_math(sample_params={"k": 1, "n": 1}),
        Metrics.avg_at_n_math(sample_params={"n": 1}),
        output_token_metric,
    ],
    version=2,
)

aime25_local = LightevalTaskConfig(
    name="aime25:local",
    prompt_function=custom_aime_prompt,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
    hf_repo="AI-MO/aimo-validation-aime-2025",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[
        Metrics.pass_at_k_math(sample_params={"k": 1, "n": 8}),
        Metrics.avg_at_n_math(sample_params={"n": 1}),
        Metrics.avg_at_n_math(sample_params={"n": 8}),
        output_token_metric,
    ],
    version=2,
)

hmmt25_local = LightevalTaskConfig(
    name="hmmt25:local",
    prompt_function=custom_aime_prompt,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
    hf_repo="MathArena/hmmt_feb_2025",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[
        Metrics.pass_at_k_math(sample_params={"k": 1, "n": 8}),
        Metrics.avg_at_n_math(sample_params={"n": 1}),
        Metrics.avg_at_n_math(sample_params={"n": 8}),
        output_token_metric,
    ],
    version=0,
)

math500_local = LightevalTaskConfig(
    name="math500:local",
    prompt_function=custom_math_500_prompt,
    sample_fields=math500_record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=model_graded_fact(),
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    generation_size=32768,
    metrics=[
        Metrics.pass_at_k_math(sample_params={"k": 1, "n": 8}),
        Metrics.avg_at_n_math(sample_params={"n": 1}),
        Metrics.avg_at_n_math(sample_params={"n": 8}),
        output_token_metric,
    ],
    version=2,
)


TASKS_TABLE = (
    [aime24_local, aime25_local, hmmt25_local, math500_local]

)

