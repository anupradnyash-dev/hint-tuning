#!/usr/bin/env python3
"""
Merge think / instruct / prefix-search results into the final SFT dataset.

Inputs
------
  think_file    : think_results.json        (Step 1 output — all 1000 problems)
  grading_file  : llm_grading_think.json    (Step 2 output — which think traces are correct)
  instruct_file : instruct_results.json     (Step 2b output — instruct baseline, all 1000)
  prefix_file   : k_prefix.json            (Step 3 output — hard problems only)

Output
------
  Alpaca-format JSON ready for SFT training.

Usage
-----
  python merge.py \
      --think    output/think_results.json \
      --grading  output/llm_grading_think.json \
      --instruct output/instruct_results.json \
      --prefix   output/k_prefix.json \
      --output   data/hint_tuning_1k.json
"""

import json
import random
import argparse

NO_HINT_PREFIXES = [
    "Let me think.",
    "Hmm,",
    "OK,",
    "Let's see.",
    "",
]
SPARSE_HINT_PREFIX = "I may need some deep thinking. "
FULL_HINT_PREFIX   = (
    "This is a complex or challenging question, and it is difficult to provide "
    "a direct answer. I need to deep think about it."
)


def merge_example(think_ex, instruct_ex, prefix_ex):
    """
    Assign a reasoning state to one problem and return a merged record.

    State 1 – No-Hint:     think model already solved it → use instruct response
    State 2 – Sparse-Hint: minimal prefix unlocked the answer (k>0, success)
    State 3 – Full-Hint:   no prefix worked; fall back to full thinking trace
    """
    question = think_ex["problem"]

    if prefix_ex is None:
        # Think model solved this problem (State 1): use the instruct response
        # so the SFT target stays concise and hint-free.
        thinking = random.choice(NO_HINT_PREFIXES)
        answer   = instruct_ex["model_response"]

    elif prefix_ex["success"]:
        thinking = SPARSE_HINT_PREFIX + prefix_ex["prefix"]
        answer   = prefix_ex["generated_text"]

    else:
        # No prefix worked — use the full thinking-model trace
        response  = think_ex["model_response"]
        think_end = response.find("</think>")
        raw_think = response[:think_end]
        if "<think>" in raw_think:
            raw_think = raw_think[raw_think.find("<think>") + 7:]
        thinking = FULL_HINT_PREFIX + raw_think
        answer   = response[think_end + len("</think>"):].strip()

    return {"question": question, "thinking": thinking, "answer": answer}


def format_example(ex):
    """Convert merged record to Alpaca SFT format."""
    output = f"<think>\n{ex['thinking'].strip()}\n</think>\n\n{ex['answer'].strip()}"
    return {"instruction": ex["question"].strip(), "input": "", "output": output}


def main():
    parser = argparse.ArgumentParser(description="Merge construction outputs into SFT dataset")
    parser.add_argument("--think",    required=True, help="think_results.json (Step 1)")
    parser.add_argument("--grading",  required=True, help="llm_grading_think.json (Step 2)")
    parser.add_argument("--instruct", required=True, help="instruct_results.json (Step 2b)")
    parser.add_argument("--prefix",   required=True, help="k_prefix.json (Step 3)")
    parser.add_argument("--output",   required=True, help="Output SFT JSON path")
    args = parser.parse_args()

    with open(args.think)    as f: think_data    = json.load(f)
    with open(args.grading)  as f: grading_data  = json.load(f)
    with open(args.instruct) as f: instruct_data = json.load(f)
    with open(args.prefix)   as f: prefix_results = json.load(f)["results"]

    # Build a dict: original problem index → prefix result (hard problems only)
    prefix_by_idx = {}
    for entry in prefix_results:
        idx = entry.get("_original_idx", entry.get("index"))
        if idx is not None:
            prefix_by_idx[idx] = entry

    # Build set of indices where think model was correct (State 1)
    think_correct = {
        item.get("question_id", i)
        for i, item in enumerate(grading_data)
        if item.get("is_correct", False)
    }

    print(f"Loaded: think={len(think_data)}  instruct={len(instruct_data)}  "
          f"prefix={len(prefix_results)}  think_correct={len(think_correct)}")

    if len(think_data) != len(instruct_data):
        raise ValueError(
            f"think ({len(think_data)}) and instruct ({len(instruct_data)}) "
            "must have the same number of entries"
        )

    merged = []
    for i, (think_ex, instruct_ex) in enumerate(zip(think_data, instruct_data)):
        if i in think_correct:
            prefix_ex = None          # State 1: think model solved it
        else:
            prefix_ex = prefix_by_idx.get(i)
            if prefix_ex is None:
                print(f"  [!] index {i}: not in think_correct and no prefix result — skipping")
                continue
        merged.append(merge_example(think_ex, instruct_ex, prefix_ex))

    formatted = [format_example(ex) for ex in merged]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)

    state1 = sum(1 for ex in merged if ex["thinking"] in NO_HINT_PREFIXES or ex["thinking"] == "")
    state2 = sum(1 for ex in merged if ex["thinking"].startswith(SPARSE_HINT_PREFIX))
    state3 = sum(1 for ex in merged if ex["thinking"].startswith(FULL_HINT_PREFIX))
    print(f"\nState distribution:")
    print(f"  State 1 – No-Hint     : {state1} ({state1/len(merged)*100:.1f}%)")
    print(f"  State 2 – Sparse-Hint : {state2} ({state2/len(merged)*100:.1f}%)")
    print(f"  State 3 – Full-Hint   : {state3} ({state3/len(merged)*100:.1f}%)")
    print(f"\nSaved {len(formatted)} samples to {args.output}")


if __name__ == "__main__":
    main()
