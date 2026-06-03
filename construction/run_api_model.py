"""
Run prefix search using a single OpenAI-compatible API model (no GPU/vLLM required).

Usage:
  python run_api_model.py \\
      --think-results  output/think_results.json \\
      --output-dir     output/prefix_search \\
      --api-model      gpt-4o \\
      --api-key        YOUR_API_KEY \\
      --base-url       https://api.openai.com/v1
"""

import argparse
import json
import os
import re
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI


# ─── utilities ───────────────────────────────────────────────────────────────

def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─── LLM-as-judge grader ────────────────────────────────────────────────────

# Grading prompt from the s1 paper (https://arxiv.org/abs/2501.19393).
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


# ─── text processing ─────────────────────────────────────────────────────────

_REFLECTION_KW = [
    "wait", "actually", "let me reconsider",
    "on second thought", "hold on", "let me rethink",
]

class TextProcessor:
    def __init__(self, keywords: List[str]):
        pattern  = "|".join(re.escape(k) for k in keywords)
        self._re = re.compile(pattern, re.IGNORECASE)

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


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--think-results",  required=True,
                   help="think_results.json from Step 1")
    p.add_argument("--think-grading",  default=None,
                   help="llm_grading_think.json — only process is_correct=False problems")
    p.add_argument("--output-dir",     required=True)
    p.add_argument("--api-model",      default="gpt-4o")
    p.add_argument("--api-key",        required=True,
                   help="API key for the instruct model")
    p.add_argument("--base-url",       default="https://api.openai.com/v1")
    p.add_argument("--instruct-rpm",   type=int, default=30)
    p.add_argument("--grader-model",   default="gpt-4o-mini",
                   help="Model used for LLM-as-judge grading")
    p.add_argument("--grader-rpm",     type=int, default=30)
    p.add_argument("--max-tokens",     type=int, default=16384)
    p.add_argument("--temperature",    type=float, default=0.7)
    p.add_argument("--max-problems",   type=int, default=None)
    p.add_argument("--resume",         action="store_true",
                   help="Resume from an existing output JSONL")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# API client (rate-limited)
# ─────────────────────────────────────────────────────────────────────────────

class RateLimitedAPIModel:
    def __init__(self, api_key, base_url, model, rpm, max_tokens, temperature):
        self.client      = OpenAI(api_key=api_key, base_url=base_url)
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self._gap        = 60.0 / max(rpm, 1)
        self._last_t     = 0.0

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

        resp = self.client.chat.completions.create(
            model=self.model, messages=messages,
            stream=False, max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        self._last_t = time.time()
        text  = resp.choices[0].message.content or ""
        usage = resp.usage
        return {
            "generated_text":   text,
            "generated_tokens": usage.completion_tokens if usage else 0,
            "prompt_tokens":    usage.prompt_tokens     if usage else 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Core: find minimal k
# ─────────────────────────────────────────────────────────────────────────────

def _extract_think_content(model_response: str) -> str:
    text = model_response
    if "<think>" in text:
        text = text[text.index("<think>") + len("<think>"):]
    if "</think>" in text:
        text = text[:text.rfind("</think>")]
    return text.strip()


def find_minimal_k(item, api_model, text_processor, grader):
    reasoning = _extract_think_content(item["model_response"])
    parts     = text_processor.split_by_reflection(reasoning)
    if len(parts) > 40:
        parts = parts[:40]
    prefixes  = create_cumulative_prefixes(parts)

    all_k = []
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

    return {
        "success":       False,
        "total_parts":   len(parts),
        "reason":        "No prefix led to correct answer",
        "all_k_results": all_k,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_jsonl(path, idx, data):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "result", "index": idx, "data": data},
                            ensure_ascii=False) + "\n")

def _jsonl_to_json(jsonl_path, json_path):
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

def print_statistics(results):
    total      = len(results)
    successful = [r for r in results if r.get("success")]
    failed     = [r for r in results if not r.get("success")]
    print(f"\n{'='*60}")
    print(f"  STATISTICS  {len(successful)}/{total} solved  "
          f"({len(successful)/total*100:.1f}%)")
    if successful:
        k_vals = [r["k_value"] for r in successful]
        k0     = sum(1 for k in k_vals if k == 0)
        k_pos  = [k for k in k_vals if k > 0]
        print(f"  k=0 (no prefix needed): {k0}  ({k0/total*100:.1f}%)")
        if k_pos:
            print(f"  k>0 (prefix helped)  : {len(k_pos)}  "
                  f"mean={np.mean(k_pos):.2f}  median={np.median(k_pos):.1f}  "
                  f"min={min(k_pos)}  max={max(k_pos)}")
        itoks = [r["instruct_tokens"] for r in successful]
        print(f"  instruct tokens: mean={np.mean(itoks):.0f}  "
              f"median={np.median(itoks):.0f}")
    print(f"  failed: {len(failed)}  ({len(failed)/total*100:.1f}%)")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    think_results = load_json(args.think_results)

    if args.think_grading:
        grading  = load_json(args.think_grading)
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
        print(f"grading filter: {before} → {len(think_results)} "
              f"(kept is_correct=False)")

    if args.max_problems:
        think_results = think_results[:args.max_problems]

    jsonl_path      = os.path.join(args.output_dir, "k_prefix.jsonl")
    slim_jsonl_path = os.path.join(args.output_dir, "k_prefix_slim.jsonl")
    done_indices    = set()

    if args.resume and os.path.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("type") == "result":
                    done_indices.add(obj["index"])
        print(f"resume: {len(done_indices)} results already done, skipping")
    else:
        metadata = {
            "type":           "metadata",
            "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
            "instruct_model": args.api_model,
            "num_problems":   len(think_results),
        }
        for p in (jsonl_path, slim_jsonl_path):
            with open(p, "w", encoding="utf-8") as f:
                f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    api_model = RateLimitedAPIModel(
        api_key     = args.api_key,
        base_url    = args.base_url,
        model       = args.api_model,
        rpm         = args.instruct_rpm,
        max_tokens  = args.max_tokens,
        temperature = args.temperature,
    )
    grader = LLMGrader(
        api_key  = args.api_key,
        base_url = args.base_url,
        model    = args.grader_model,
        rpm      = args.grader_rpm,
    )
    text_processor = TextProcessor(_REFLECTION_KW)

    results = []
    t0      = time.time()

    for i, item in enumerate(think_results):
        orig_idx = item.get("_original_idx", i)

        if orig_idx in done_indices:
            continue

        if i % 10 == 0:
            elapsed   = time.time() - t0
            remaining = len(think_results) - len(done_indices) - len(results)
            eta       = elapsed / max(len(results), 1) * remaining
            print(f"[{i+1}/{len(think_results)}]  "
                  f"elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m")

        try:
            result = find_minimal_k(item, api_model, text_processor, grader)
            result.update({"problem": item["problem"],
                            "gold_answer": item["gold_answer"]})
            results.append(result)

            if result.get("success"):
                k   = result["k_value"]
                tot = result.get("total_parts", "?")
                tok = result.get("instruct_tokens", 0)
                print(f"  [{orig_idx:4d}] ✓  k={k}/{tot}  tokens={tok}")
            else:
                print(f"  [{orig_idx:4d}] ✗  {result.get('reason', 'failed')}")

            _write_jsonl(jsonl_path, orig_idx, result)
            slim = {k: v for k, v in result.items()
                    if k not in ("generated_text", "prefix", "problem")}
            _write_jsonl(slim_jsonl_path, orig_idx, slim)

        except Exception as e:
            print(f"  [{orig_idx:4d}] ERROR: {e}")
            traceback.print_exc()
            with open(jsonl_path, "a") as f:
                f.write(json.dumps({"type": "error", "index": orig_idx,
                                    "error": str(e)}, ensure_ascii=False) + "\n")

    _jsonl_to_json(jsonl_path,      os.path.join(args.output_dir, "k_prefix.json"))
    _jsonl_to_json(slim_jsonl_path, os.path.join(args.output_dir, "k_prefix_slim.json"))

    elapsed = time.time() - t0
    print(f"\nDone.  elapsed={elapsed/60:.1f} min")
    print_statistics(results)


if __name__ == "__main__":
    main()
