"""
Four-step data construction pipeline:

  Step 1  think  – Think model generates full reasoning traces.
  Step 2  grade  – LLM-as-judge decides which traces are correct.
                   Problems the think model already solved → State 1 (No-Hint).
                   The rest go to Step 3.
  Step 3  prefix – For each unsolved problem, find the minimal prefix K
                   of the thinking trace that lets an instruct model answer
                   correctly (State 2 – Sparse-Hint, or State 3 – Full-Hint
                   if no prefix works).
  Step 4  merge  – Assemble the three states into Alpaca SFT format
                   (see merge.py).

Usage:
  # Step 1: generate think traces
  python pipeline.py --mode think \\
      --think-model Qwen/Qwen3-4B-Thinking-2507 \\
      --dataset     data/problems.json \\
      --output-dir  output/

  # Step 2: LLM-as-judge grading of think traces
  python pipeline.py --mode grade \\
      --think-results output/think_results.json \\
      --instruct-models-config construction/instruct_models.yaml \\
      --output-dir    output/

  # Step 3: minimal-prefix search (only on problems think model got wrong)
  python pipeline.py --mode prefix \\
      --think-results  output/think_results.json \\
      --think-grading  output/llm_grading_think.json \\
      --instruct-models-config construction/instruct_models.yaml \\
      --output-dir     output/
"""

import argparse
import gc
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


# ─── utilities ───────────────────────────────────────────────────────────────

def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_json(data: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

# ─── LLM-as-judge grader ────────────────────────────────────────────────────

# Grading prompt from the s1 paper (https://arxiv.org/abs/2501.19393).
# Used instead of rule-based checking because s1K contains non-math problems
# whose answers are not always in strict \boxed{} format.
_GRADING_SYSTEM = (
    "You are an AI assistant for grading a science problem. "
    "The user will provide you with the question itself, an attempt made by "
    "a student and the correct answer to the problem. Your job is to judge "
    "whether the attempt is correct by comparing it with the correct answer. "
    "If the expected solution concludes with a number or choice, there should "
    "be no ambiguity. If the expected solution involves going through the "
    "entire reasoning process, you should judge the attempt based on whether "
    "the reasoning process is correct with correct answer if helpful.\n"
    "The user will provide the attempt and the correct answer in the "
    "following format:\n"
    "# Problem\n{problem}\n"
    "## Attempt\n{attempt}\n"
    "## Correct answer\n{solution}\n"
    'Explain your reasoning, and end your response on a new line with only '
    '"Yes" or "No" (without quotes).'
)

def _grading_user_msg(problem: str, gold_answer: str, prediction: str) -> str:
    return (
        f"# Problem\n{problem}\n"
        f"## Attempt\n{prediction}\n"
        f"## Correct answer\n{gold_answer}"
    )


class LLMGrader:
    """Rate-limited LLM-as-judge grader via OpenAI-compatible API."""

    def __init__(self, api_key: str, base_url: str, model: str, rpm: int = 30):
        from openai import OpenAI
        self.client    = OpenAI(api_key=api_key, base_url=base_url)
        self.model     = model
        self.delay     = 60.0 / max(rpm, 1)
        self.last_call = 0.0

    def grading_answer(self, problem: str, gold_answer: str,
                       prediction: str) -> bool:
        elapsed = time.time() - self.last_call
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

        messages = [
            {"role": "system", "content": _GRADING_SYSTEM},
            {"role": "user",   "content": _grading_user_msg(problem, gold_answer, prediction)},
        ]
        for attempt in range(2):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    max_tokens=512, temperature=0.0,
                )
                self.last_call = time.time()
                return (resp.choices[0].message.content or "").strip().lower().endswith("yes")
            except Exception as e:
                if attempt == 0:
                    print(f"Grading error (retry): {e}"); time.sleep(2)
                else:
                    print(f"Grading error (giving up): {e}"); return False


class LocalLLMGrader(LLMGrader):
    """Grader backed by a local vLLM server.

    Disables Qwen3 thinking mode via extra_body so grading stays cheap.
    """

    def grading_answer(self, problem: str, gold_answer: str,
                       prediction: str) -> bool:
        elapsed = time.time() - self.last_call
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

        messages = [
            {"role": "system", "content": _GRADING_SYSTEM},
            {"role": "user",   "content": _grading_user_msg(problem, gold_answer, prediction)},
        ]
        for attempt in range(2):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    max_tokens=512, temperature=0.0,
                    extra_body={"enable_thinking": False},
                )
                self.last_call = time.time()
                return (resp.choices[0].message.content or "").strip().lower().endswith("yes")
            except Exception as e:
                if attempt == 0:
                    print(f"Grading error (retry): {e}"); time.sleep(2)
                else:
                    print(f"Grading error (giving up): {e}"); return False


# ─── text processing ─────────────────────────────────────────────────────────

class TextProcessor:
    """Split a thinking trace at reflection-keyword boundaries."""

    def __init__(self, keywords: List[str]):
        pattern    = "|".join(re.escape(k) for k in keywords)
        self._re   = re.compile(pattern, re.IGNORECASE)

    def split_by_reflection(self, text: str) -> List[str]:
        parts, last = [], 0
        for m in self._re.finditer(text):
            if m.start() > last:
                parts.append(text[last:m.start()])
            last = m.start()
        parts.append(text[last:])
        return [p for p in parts if p.strip()]


def create_cumulative_prefixes(parts: List[str]) -> List[str]:
    """["A","B","C"] → ["", "A", "AB", "ABC"]  (prefixes[0] is always empty)."""
    prefixes = [""]
    for i in range(1, len(parts) + 1):
        prefixes.append("".join(parts[:i]))
    return prefixes


# ─── local vLLM model inference ──────────────────────────────────────────────

class ModelInference:
    """Thin vLLM wrapper used for both the think model (Step 1) and local
    instruct models (Step 2).

    Prefix injection uses ``continue_final_message=True`` so the model
    continues an already-started assistant turn (the partial thinking trace).
    If your tokenizer does not support that flag, set
    ``HINT_TUNING_LEGACY_PREFIX=1`` in the environment to fall back to
    appending the prefix to the raw prompt string.
    """

    def __init__(self, config: Dict[str, Any], model_type: str = "base",
                 enable_thinking: Optional[bool] = None):
        from vllm import LLM, SamplingParams

        if model_type == "reasoning":
            model_path = config["models"]["reasoning_model"]
        else:
            model_path = config["models"]["base_model"]

        gpu_cfg = config.get("gpu", {})
        inf_cfg = config.get("inference", {})

        self._llm = LLM(
            model=model_path,
            tensor_parallel_size=gpu_cfg.get("tensor_parallel_size", 4),
            max_model_len=inf_cfg.get("max_model_len", 32768),
            trust_remote_code=True,
        )
        self._sampling = SamplingParams(
            temperature=inf_cfg.get("temperature", 0.7),
            top_p=inf_cfg.get("top_p", 0.95),
            max_tokens=inf_cfg.get("max_tokens", 16384),
        )
        self._tokenizer       = self._llm.get_tokenizer()
        self._enable_thinking = enable_thinking
        self._legacy_prefix   = bool(os.environ.get("HINT_TUNING_LEGACY_PREFIX"))

    def _make_prompt(self, problem: str, prefix: str) -> str:
        messages: List[Dict[str, str]] = [{"role": "user", "content": problem}]
        kwargs: Dict[str, Any] = {"tokenize": False}

        if prefix:
            messages.append({"role": "assistant", "content": prefix})
            if self._legacy_prefix:
                # Older tokenizers: build base prompt, append prefix manually.
                base = self._tokenizer.apply_chat_template(
                    [messages[0]], tokenize=False, add_generation_prompt=True,
                )
                return base + prefix
            kwargs["continue_final_message"] = True
            kwargs["add_generation_prompt"]  = False
        else:
            kwargs["add_generation_prompt"] = True

        if self._enable_thinking is not None:
            kwargs["enable_thinking"] = self._enable_thinking

        return self._tokenizer.apply_chat_template(messages, **kwargs)

    def batch_generate(self, problems: List[str], prefixes: List[str],
                       return_tokens: bool = True) -> List[Dict[str, Any]]:
        prompts = [self._make_prompt(p, pf) for p, pf in zip(problems, prefixes)]
        outputs = self._llm.generate(prompts, self._sampling)
        results = []
        for out in outputs:
            o = out.outputs[0]
            results.append({
                "generated_text":   o.text,
                "generated_tokens": len(o.token_ids),
                "prompt_tokens":    len(out.prompt_token_ids),
                "total_tokens":     len(out.prompt_token_ids) + len(o.token_ids),
            })
        return results


# ─── API model inference ─────────────────────────────────────────────────────

class APIModelInference:
    """OpenAI-compatible API model for Step 2 prefix search (no GPU required)."""

    def __init__(self, model_cfg: Dict[str, Any]):
        from openai import OpenAI
        self._client      = OpenAI(api_key=model_cfg["api_key"],
                                   base_url=model_cfg["base_url"])
        self._model       = model_cfg["api_model"]
        self._max_tokens  = model_cfg.get("max_tokens", 16384)
        self._temperature = model_cfg.get("temperature", 0.7)
        self._gap         = 60.0 / max(model_cfg.get("rpm", 30), 1)
        self._last_t      = 0.0

    def generate_one(self, problem: str, prefix: str = "") -> Dict[str, Any]:
        wait = self._gap - (time.time() - self._last_t)
        if wait > 0:
            time.sleep(wait)

        if prefix:
            messages = [
                {"role": "user",      "content": problem},
                {"role": "assistant", "content": prefix},
                {"role": "user",      "content":
                    "Continue your reasoning from where you left off "
                    "and provide the final answer within \\boxed{}."},
            ]
        else:
            messages = [{"role": "user", "content": problem}]

        resp = self._client.chat.completions.create(
            model=self._model, messages=messages,
            stream=False, max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        self._last_t = time.time()
        text  = resp.choices[0].message.content or ""
        usage = resp.usage
        return {
            "generated_text":   text,
            "generated_tokens": usage.completion_tokens if usage else 0,
            "prompt_tokens":    usage.prompt_tokens     if usage else 0,
        }

    def batch_generate(self, problems: List[str], prefixes: List[str],
                       return_tokens: bool = True) -> List[Dict[str, Any]]:
        return [self.generate_one(p, pf) for p, pf in zip(problems, prefixes)]


# ─────────────────────────────────────────────────────────────────────────────

try:
    from datasets import load_from_disk
except ImportError:
    pass  # only needed for Step 1


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

_HERE = os.path.abspath(__file__)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Think→Split→Instruct minimal-prefix pipeline (multi-model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--output-dir",   type=str, required=True)
    p.add_argument("--config",       type=str,
                   default=os.path.join(os.path.dirname(_HERE), "config.yaml"))
    p.add_argument("--mode",         choices=["think", "grade", "instruct", "prefix"],
                   required=True)
    p.add_argument("--max-problems", type=int, default=None)

    p.add_argument("--think-model",  type=str)
    p.add_argument("--dataset",      type=str)

    p.add_argument("--think-results", type=str)
    p.add_argument("--think-grading", type=str, default=None,
                   help="llm_grading_think.json — only problems with "
                        "is_correct=False are processed")

    p.add_argument("--instruct-models-config", type=str,
                   default=os.path.join(os.path.dirname(_HERE),
                                        "instruct_models.yaml"))
    p.add_argument("--instruct-model",       type=str)
    p.add_argument("--tensor-parallel-size", type=int, default=None)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Think model generation
# ─────────────────────────────────────────────────────────────────────────────

def step1_generate_think(
    model_path:   str,
    config:       Dict[str, Any],
    dataset,
    output_path:  str,
    max_problems: Optional[int] = None,
) -> List[Dict[str, Any]]:
    _header("Step 1 · Think model generation")
    print(f"  model    : {model_path}")

    cfg   = _override_model(config, reasoning_model=model_path)
    model = ModelInference(cfg, model_type="reasoning")

    problems = list(dataset)
    if max_problems:
        problems = problems[:max_problems]
    print(f"  problems : {len(problems)}")

    generation_results = model.batch_generate(
        problems=[p["question"] for p in problems],
        prefixes=[""] * len(problems),
        return_tokens=True,
    )

    results = []
    for prob_data, gen in zip(problems, generation_results):
        results.append({
            "problem":        prob_data["question"],
            "gold_answer":    prob_data["solution"],
            "model_response": gen["generated_text"],
            "metrics": {
                "prompt_tokens":   gen["prompt_tokens"],
                "solution_tokens": gen["generated_tokens"],
                "total_tokens":    gen["total_tokens"],
            },
        })

    save_json(results, output_path)

    total   = len(results)
    avg_tok = sum(r["metrics"]["solution_tokens"] for r in results) / total
    print(f"\n  problems   : {total}")
    print(f"  avg tokens : {avg_tok:.0f}")
    print(f"  saved to   : {output_path}")
    print(f"  → run --mode grade next to evaluate correctness with LLM-as-judge")

    del model
    gc.collect()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – LLM-as-judge grading of think traces
# ─────────────────────────────────────────────────────────────────────────────

def step2_grade_think(
    think_results: List[Dict[str, Any]],
    grader:        LLMGrader,
    output_path:   str,
) -> List[Dict[str, Any]]:
    """Grade each think-model response with LLM-as-judge.

    Produces llm_grading_think.json: [{question_id, is_correct}, ...].
    Problems where is_correct=False are passed to Step 3 (prefix search).
    s1K contains non-math problems, so rule-based verification is insufficient.
    """
    _header("Step 2 · LLM-as-judge grading of think traces")
    print(f"  problems : {len(think_results)}")

    results = []
    for idx, item in enumerate(think_results):
        if idx % 50 == 0:
            print(f"  [{idx}/{len(think_results)}]")
        is_correct = grader.grading_answer(
            item["problem"], item["gold_answer"], item["model_response"])
        results.append({"question_id": idx, "is_correct": is_correct})

    save_json(results, output_path)

    n_correct = sum(1 for r in results if r["is_correct"])
    print(f"\n  think model accuracy : {n_correct}/{len(results)} "
          f"= {n_correct/len(results)*100:.1f}%")
    print(f"  → {len(results) - n_correct} problems pass to prefix search")
    print(f"  saved to : {output_path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 2b – Instruct model baseline (all problems, no prefix)
# ─────────────────────────────────────────────────────────────────────────────

def step2b_generate_instruct(
    model_path:   str,
    config:       Dict[str, Any],
    think_results: List[Dict[str, Any]],
    output_path:  str,
    max_problems: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run the instruct model on every problem with no prefix.

    Output is instruct_results.json, used by merge.py for State-1 samples
    (problems the think model already solved, where we want a shorter,
    hint-free instruct response as the SFT target).
    """
    _header("Step 2b · Instruct model baseline (no prefix)")
    print(f"  model    : {model_path}")

    cfg   = _override_model(config, base_model=model_path)
    model = ModelInference(cfg, model_type="base", enable_thinking=False)

    problems = think_results if not max_problems else think_results[:max_problems]
    print(f"  problems : {len(problems)}")

    generation_results = model.batch_generate(
        problems=[p["problem"] for p in problems],
        prefixes=[""] * len(problems),
        return_tokens=True,
    )

    results = []
    for item, gen in zip(problems, generation_results):
        results.append({
            "problem":        item["problem"],
            "gold_answer":    item["gold_answer"],
            "model_response": gen["generated_text"],
            "metrics": {
                "prompt_tokens":   gen["prompt_tokens"],
                "solution_tokens": gen["generated_tokens"],
                "total_tokens":    gen["total_tokens"],
            },
        })

    save_json(results, output_path)

    avg_tok = sum(r["metrics"]["solution_tokens"] for r in results) / len(results)
    print(f"\n  problems   : {len(results)}")
    print(f"  avg tokens : {avg_tok:.0f}")
    print(f"  saved to   : {output_path}")
    print(f"  → run --mode prefix next to find minimal hint prefixes")

    del model
    gc.collect()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Minimal-prefix search  (multi-model)
# ─────────────────────────────────────────────────────────────────────────────

def step3_all_models(
    model_cfgs:      List[Dict[str, Any]],
    base_config:     Dict[str, Any],
    think_results:   List[Dict[str, Any]],
    base_output_dir: str,
    grader:          LLMGrader,
) -> None:
    _header(f"Step 3 · Prefix search over {len(model_cfgs)} instruct model(s)")
    print(f"  problems  : {len(think_results)}")
    for i, mc in enumerate(model_cfgs):
        print(f"  [{i+1}] {mc['name']}  ({mc['type']})")

    for model_cfg in model_cfgs:
        name       = model_cfg["name"]
        model_type = model_cfg["type"]
        out_dir    = os.path.join(base_output_dir, name)
        os.makedirs(out_dir, exist_ok=True)

        _header(f"  Model: {name}  ({model_type})")

        if model_type == "local":
            _run_local_model(model_cfg, base_config, think_results, out_dir, grader)
        elif model_type == "api":
            _run_api_model(model_cfg, think_results, out_dir, grader,
                           base_config.get("reflection_keywords", _DEFAULT_KW))
        else:
            print(f"  [!] Unknown model type '{model_type}', skipping.")


def _run_local_model(
    model_cfg:     Dict[str, Any],
    base_config:   Dict[str, Any],
    think_results: List[Dict[str, Any]],
    out_dir:       str,
    grader:        LLMGrader,
) -> None:
    cfg = dict(base_config)
    cfg["models"]   = {"base_model": model_cfg["path"]}
    cfg["gpu"]      = dict(base_config.get("gpu", {}))
    cfg["gpu"]["tensor_parallel_size"] = model_cfg.get(
        "tensor_parallel_size", cfg["gpu"].get("tensor_parallel_size", 4))
    cfg["inference"] = dict(base_config.get("inference", {}))
    cfg["inference"]["max_model_len"] = model_cfg.get(
        "max_model_len", cfg["inference"].get("max_model_len", 32768))
    if "max_new_tokens" in model_cfg:
        cfg["inference"]["max_tokens"] = model_cfg["max_new_tokens"]

    enable_thinking = model_cfg.get("enable_thinking", None)
    instruct_model  = ModelInference(cfg, model_type="base",
                                     enable_thinking=enable_thinking)
    text_processor  = TextProcessor(
        base_config.get("reflection_keywords", _DEFAULT_KW))

    _run_prefix_search(
        instruct_model  = instruct_model,
        text_processor  = text_processor,
        think_results   = think_results,
        out_dir         = out_dir,
        grader          = grader,
        model_label     = model_cfg["path"],
        sequential_mode = False,
    )

    del instruct_model
    gc.collect()
    print(f"  GPU memory released after {model_cfg['name']}")


def _run_api_model(
    model_cfg:          Dict[str, Any],
    think_results:      List[Dict[str, Any]],
    out_dir:            str,
    grader:             LLMGrader,
    reflection_keywords: List[str],
) -> None:
    api_model      = APIModelInference(model_cfg)
    text_processor = TextProcessor(reflection_keywords)

    _run_prefix_search(
        instruct_model  = api_model,
        text_processor  = text_processor,
        think_results   = think_results,
        out_dir         = out_dir,
        grader          = grader,
        model_label     = model_cfg["api_model"],
        sequential_mode = True,
    )


def _run_prefix_search(
    instruct_model,
    text_processor:  TextProcessor,
    think_results:   List[Dict[str, Any]],
    out_dir:         str,
    grader:          LLMGrader,
    model_label:     str,
    sequential_mode: bool,
    resume:          bool = True,
) -> List[Dict[str, Any]]:
    jsonl_path      = os.path.join(out_dir, "k_prefix.jsonl")
    slim_jsonl_path = os.path.join(out_dir, "k_prefix_slim.jsonl")

    done_indices: set = set()
    if resume and os.path.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if obj.get("type") == "result":
                        done_indices.add(obj["index"])
                except Exception:
                    pass
        if done_indices:
            print(f"  resume: {len(done_indices)} items already done, skipping")

    if not done_indices:
        metadata = {
            "type":            "metadata",
            "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
            "instruct_model":  model_label,
            "num_problems":    len(think_results),
            "sequential_mode": sequential_mode,
        }
        for path in (jsonl_path, slim_jsonl_path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    results: List[Dict[str, Any]] = []
    t0 = time.time()

    for i, item in enumerate(think_results):
        orig_idx = item.get("_original_idx", i)

        if orig_idx in done_indices:
            continue

        if i % 10 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / max(i, 1) * (len(think_results) - i)
            print(f"\n  [{i+1}/{len(think_results)}]  "
                  f"elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m")

        try:
            if sequential_mode:
                result = _find_minimal_k_sequential(
                    item, instruct_model, text_processor, grader)
            else:
                result = _find_minimal_k_batch(
                    item, instruct_model, text_processor, grader)

            result.update({"problem": item["problem"],
                            "gold_answer": item["gold_answer"]})
            results.append(result)
            _log_item(orig_idx, result)

            _write_jsonl(jsonl_path, orig_idx, result)
            slim = {k: v for k, v in result.items()
                    if k not in ("generated_text", "prefix", "problem")}
            _write_jsonl(slim_jsonl_path, orig_idx, slim)

        except Exception as exc:
            import traceback
            print(f"  [!] item {orig_idx} error: {exc}")
            traceback.print_exc()
            _append_line(jsonl_path,
                         {"type": "error", "index": i, "error": str(exc)})

    json_path      = os.path.join(out_dir, "k_prefix.json")
    slim_json_path = os.path.join(out_dir, "k_prefix_slim.json")
    _jsonl_to_json(jsonl_path, json_path)
    _jsonl_to_json(slim_jsonl_path, slim_json_path)

    elapsed = time.time() - t0
    print(f"\n  saved → {out_dir}  (elapsed {elapsed/60:.1f} min)")
    print_statistics(results)
    return results


def _extract_think_content(model_response: str) -> str:
    text = model_response
    if "<think>" in text:
        text = text[text.index("<think>") + len("<think>"):]
    if "</think>" in text:
        text = text[:text.rfind("</think>")]
    return text.strip()


def _find_minimal_k_batch(
    item:           Dict[str, Any],
    instruct_model: ModelInference,
    text_processor: TextProcessor,
    grader:         LLMGrader,
) -> Dict[str, Any]:
    """Local vLLM: one prefix at a time with early stopping.

    Generating all N prefixes in one batch risks vLLM's 5-min RPC timeout
    when each output reaches max_new_tokens.  Single-item calls stay short.
    """
    reasoning_text = _extract_think_content(item["model_response"])
    parts    = text_processor.split_by_reflection(reasoning_text)
    if len(parts) > 40:
        parts = parts[:40]
    prefixes = create_cumulative_prefixes(parts)

    all_k: List[Dict[str, Any]] = []
    for k, prefix in enumerate(prefixes):
        gen_list = instruct_model.batch_generate(
            problems=[item["problem"]],
            prefixes=[prefix],
            return_tokens=True,
        )
        gen = gen_list[0]
        correct = grader.grading_answer(
            item["problem"], item["gold_answer"], gen["generated_text"])
        all_k.append({"k": k, "correct": correct, "tokens": gen["generated_tokens"]})
        if correct:
            return {
                "success":         True,
                "k_value":         k,
                "total_parts":     len(parts),
                "prefix":          prefix,
                "prefix_tokens":   gen["prompt_tokens"],
                "instruct_tokens": gen["generated_tokens"],
                "generated_text":  gen["generated_text"],
                "all_k_results":   all_k,
            }

    return {"success": False, "total_parts": len(parts),
            "reason": "No prefix led to correct answer", "all_k_results": all_k}


def _find_minimal_k_sequential(
    item:           Dict[str, Any],
    api_model:      APIModelInference,
    text_processor: TextProcessor,
    grader:         LLMGrader,
) -> Dict[str, Any]:
    """API path: generate + grade one prefix at a time, stop at first success."""
    reasoning_text = _extract_think_content(item["model_response"])
    parts    = text_processor.split_by_reflection(reasoning_text)
    if len(parts) > 40:
        parts = parts[:40]
    prefixes = create_cumulative_prefixes(parts)

    all_k: List[Dict[str, Any]] = []
    for k, prefix in enumerate(prefixes):
        gen     = api_model.generate_one(item["problem"], prefix)
        correct = grader.grading_answer(
            item["problem"], item["gold_answer"], gen["generated_text"])
        all_k.append({"k": k, "correct": correct, "tokens": gen["generated_tokens"]})
        if correct:
            return {
                "success":         True,
                "k_value":         k,
                "total_parts":     len(parts),
                "prefix":          prefix,
                "prefix_tokens":   gen["prompt_tokens"],
                "instruct_tokens": gen["generated_tokens"],
                "generated_text":  gen["generated_text"],
                "all_k_results":   all_k,
            }

    return {"success": False, "total_parts": len(parts),
            "reason": "No prefix led to correct answer", "all_k_results": all_k}


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def print_statistics(results: List[Dict[str, Any]]) -> None:
    total = len(results)
    if total == 0:
        print(f"\n{'='*60}\n  STATISTICS  no results collected\n{'='*60}")
        return

    successful = [r for r in results if r.get("success")]
    failed     = [r for r in results if not r.get("success")]

    print(f"\n{'='*60}")
    print(f"  STATISTICS  ({len(successful)}/{total} solved, "
          f"{len(successful)/total*100:.1f}%)")
    print(f"{'='*60}")
    if not successful:
        return

    k_vals = [r["k_value"] for r in successful]
    k0_cnt = sum(1 for k in k_vals if k == 0)
    k_pos  = [k for k in k_vals if k > 0]

    print(f"\n  Minimal-K distribution:")
    print(f"    k = 0  (instruct alone)  : {k0_cnt:4d}  ({k0_cnt/total*100:.1f}%)")
    if k_pos:
        print(f"    k > 0  (needs prefix)    : {len(k_pos):4d}  ({len(k_pos)/total*100:.1f}%)")
        print(f"      mean={np.mean(k_pos):.2f}  median={np.median(k_pos):.1f}  "
              f"min={min(k_pos)}  max={max(k_pos)}")
        buckets = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 9999)]
        print(f"\n  K-value buckets (k>0):")
        for lo, hi in buckets:
            cnt   = sum(1 for k in k_pos if lo <= k <= hi)
            label = f"{lo}-{hi}" if hi < 9999 else f"{lo}+"
            bar   = "█" * max(1, cnt // max(1, len(k_pos) // 20))
            print(f"    k={label:6s} : {cnt:4d}  {bar}")

    print(f"\n  Failed (no prefix worked) : {len(failed):4d}  ({len(failed)/total*100:.1f}%)")
    itoks = [r["instruct_tokens"] for r in successful]
    print(f"\n  Instruct tokens (at minimal k):")
    print(f"    mean={np.mean(itoks):.0f}  median={np.median(itoks):.0f}  "
          f"min={min(itoks)}  max={max(itoks)}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_KW = [
    "wait", "actually", "let me reconsider",
    "on second thought", "hold on", "let me rethink",
]


def _override_model(config: Dict, **kwargs) -> Dict:
    cfg = dict(config)
    cfg["models"] = dict(config.get("models", {}))
    cfg["models"].update(kwargs)
    return cfg

def _header(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def _log_item(i: int, result: Dict[str, Any]) -> None:
    if result.get("success"):
        k   = result["k_value"]
        tot = result.get("total_parts", "?")
        tok = result.get("instruct_tokens", 0)
        print(f"  [{i:4d}] ✓  k={k}/{tot}  instruct_tokens={tok}")
    else:
        print(f"  [{i:4d}] ✗  {result.get('reason', 'failed')}")

def _write_jsonl(path: str, idx: int, data: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "result", "index": idx, "data": data},
                            ensure_ascii=False) + "\n")

def _append_line(path: str, obj: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _jsonl_to_json(jsonl_path: str, json_path: str) -> None:
    metadata_obj, results = {}, []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj["type"] == "metadata":
                metadata_obj = {k: v for k, v in obj.items() if k != "type"}
            elif obj["type"] == "result":
                entry = dict(obj["data"])
                entry["_original_idx"] = obj["index"]
                results.append(entry)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata_obj, "results": results},
                  f, ensure_ascii=False, indent=2)

def _load_think_results(
    think_results_path: str,
    think_grading_path: Optional[str],
    max_problems:       Optional[int],
) -> List[Dict[str, Any]]:
    think_results = load_json(think_results_path)

    if think_grading_path:
        grading  = load_json(think_grading_path)
        hard_ids = {
            item.get("question_id", idx)
            for idx, item in enumerate(grading)
            if not item.get("is_correct", True)
        }
        before        = len(think_results)
        think_results = [
            {**item, "_original_idx": idx}
            for idx, item in enumerate(think_results)
            if idx in hard_ids
        ]
        print(f"  grading filter: {before} → {len(think_results)} "
              f"(kept is_correct=False)")

    if max_problems:
        think_results = think_results[:max_problems]
    return think_results

def _load_models_config(config_path: str) -> tuple:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("models", []), cfg.get("grader", {})


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args        = parse_args()
    base_config = load_config(args.config)

    if args.think_model:
        base_config.setdefault("models", {})["reasoning_model"] = args.think_model
    if args.dataset:
        base_config.setdefault("datasets", {})["s1k-1.1"] = args.dataset
    if args.tensor_parallel_size:
        base_config.setdefault("gpu", {})["tensor_parallel_size"] = args.tensor_parallel_size

    os.makedirs(args.output_dir, exist_ok=True)

    think_results_path = (
        args.think_results
        if args.think_results
        else os.path.join(args.output_dir, "think_results.json")
    )

    # ── grader (needed by both grade and prefix modes) ───────────────────────
    if args.mode in ("grade", "prefix"):
        _, grader_cfg    = _load_models_config(args.instruct_models_config)
        _grader_base_url = grader_cfg.get("base_url", "http://127.0.0.1:8001/v1")
        _grader_cls = (
            LocalLLMGrader
            if "127.0.0.1" in _grader_base_url or "localhost" in _grader_base_url
            else LLMGrader
        )
        grader = _grader_cls(
            api_key  = grader_cfg.get("api_key",  "YOUR_API_KEY"),
            base_url = _grader_base_url,
            model    = grader_cfg.get("model",    "YOUR_GRADER_MODEL"),
            rpm      = grader_cfg.get("rpm",      30),
        )
        print(f"  grader: {_grader_cls.__name__} → {_grader_base_url}")

    # ── Step 1: generate think traces ────────────────────────────────────────
    if args.mode == "think":
        if "reasoning_model" not in base_config.get("models", {}):
            raise ValueError("--think-model is required for --mode think")
        dataset_path = base_config.get("datasets", {}).get("s1k-1.1", "data/problems.json")
        if dataset_path.endswith(".json"):
            raw     = load_json(dataset_path)
            dataset = [
                {
                    "question": r.get("problem",     r.get("question", "")),
                    "solution": r.get("gold_answer",  r.get("solution", "")),
                }
                for r in raw
            ]
        else:
            dataset = load_from_disk(dataset_path)["train"]
        step1_generate_think(
            model_path   = base_config["models"]["reasoning_model"],
            config       = base_config,
            dataset      = dataset,
            output_path  = think_results_path,
            max_problems = args.max_problems,
        )

    # ── Step 2: LLM-as-judge grading ─────────────────────────────────────────
    elif args.mode == "grade":
        think_results     = load_json(think_results_path)
        grading_out_path  = os.path.join(args.output_dir, "llm_grading_think.json")
        step2_grade_think(
            think_results = think_results,
            grader        = grader,
            output_path   = grading_out_path,
        )

    # ── Step 2b: instruct model baseline (all problems, no prefix) ────────────
    elif args.mode == "instruct":
        if not args.instruct_model:
            model_cfgs, _ = _load_models_config(args.instruct_models_config)
            local_cfgs    = [m for m in model_cfgs if m.get("type") == "local"]
            if not local_cfgs:
                raise ValueError(
                    "--instruct-model is required (no local model found in "
                    f"{args.instruct_models_config})"
                )
            instruct_model_path = local_cfgs[0]["path"]
        else:
            instruct_model_path = args.instruct_model

        think_results     = load_json(think_results_path)
        instruct_out_path = os.path.join(args.output_dir, "instruct_results.json")
        step2b_generate_instruct(
            model_path    = instruct_model_path,
            config        = base_config,
            think_results = think_results,
            output_path   = instruct_out_path,
            max_problems  = args.max_problems,
        )

    # ── Step 3: minimal-prefix search ────────────────────────────────────────
    elif args.mode == "prefix":
        think_results = _load_think_results(
            think_results_path = think_results_path,
            think_grading_path = args.think_grading,
            max_problems       = args.max_problems,
        )
        if args.instruct_model:
            model_cfgs = [{
                "name": os.path.basename(args.instruct_model),
                "type": "local",
                "path": args.instruct_model,
                "tensor_parallel_size": (
                    args.tensor_parallel_size
                    or base_config.get("gpu", {}).get("tensor_parallel_size", 4)
                ),
                "max_model_len": base_config.get("inference", {}).get(
                    "max_model_len", 32768),
            }]
        else:
            model_cfgs, _ = _load_models_config(args.instruct_models_config)

        step3_all_models(
            model_cfgs      = model_cfgs,
            base_config     = base_config,
            think_results   = think_results,
            base_output_dir = args.output_dir,
            grader          = grader,
        )


if __name__ == "__main__":
    main()
