#!/usr/bin/env bash
# Evaluate a single model with lighteval (custom_tasks.py, all tasks).
# Usage:
#   bash eval.sh <model_path_or_hf_id> [output_dir]
#
# Examples:
#   bash eval.sh Qwen/hint-tuning-7b
#   bash eval.sh ./my_checkpoint output/my_results
#   CUDA_VISIBLE_DEVICES=4,5,6,7 bash eval.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
#
# AIME24/25, HMMT25, MATH-500 are natively supported by lighteval (auto-download).
# See https://github.com/huggingface/lighteval for the exact task names to use.

set -euo pipefail

MODEL_PATH="${1:?Usage: bash eval.sh <model_path_or_hf_id> [output_dir]}"
OUT_DIR="${2:-./eval_results}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUSTOM_TASKS="${SCRIPT_DIR}/custom_tasks.py"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export EVAL_MODEL_PATH="${MODEL_PATH}"

TASKS="aime24:local,aime25:local,hmmt25:local,math500:local"

# ─── helpers ─────────────────────────────────────────────────────────────────

# Some SFT training frameworks store an incorrect tokenizer_class in
# tokenizer_config.json. This function repairs the field in-place when needed.
fix_tokenizer_class_if_needed() {
  local ckpt_dir="$1"
  python3 - "$ckpt_dir" <<'PY'
import json, sys
from pathlib import Path

ckpt = Path(sys.argv[1])
tok_path = ckpt / "tokenizer_config.json"
cfg_path = ckpt / "config.json"
if not tok_path.is_file():
    sys.exit(0)

with open(tok_path, encoding="utf-8") as f:
    tok_cfg = json.load(f)
cur = tok_cfg.get("tokenizer_class", "")
if cur not in ("TokenizersBackend", "TokenizersBackendFast", ""):
    sys.exit(0)

arch = ""
if cfg_path.is_file():
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    archs = cfg.get("architectures") or []
    arch = archs[0] if archs else ""

if "Qwen2" in arch or "Qwen3" in arch:
    fixed = "Qwen2Tokenizer"
elif "Llama" in arch or "Mistral" in arch:
    fixed = "LlamaTokenizer"
else:
    fixed = "Qwen2Tokenizer"

tok_cfg["tokenizer_class"] = fixed
with open(tok_path, "w", encoding="utf-8") as f:
    json.dump(tok_cfg, f, indent=2, ensure_ascii=False)
    f.write("\n")
print(f"  [fix] tokenizer_class: {cur!r} -> {fixed!r}  (arch={arch!r})")
PY
}

write_model_yaml() {
  local model_path="$1"
  local out_yaml="$2"
  mkdir -p "$(dirname "$out_yaml")"
  cat > "$out_yaml" <<EOF
model_parameters:
  model_name: "${model_path}"
  dtype: "bfloat16"
  tensor_parallel_size: 4
  data_parallel_size: 1
  gpu_memory_utilization: 0.95
  max_model_length: 32768
  trust_remote_code: true
  override_chat_template: true
  generation_parameters:
    temperature: 0.6
    top_p: 0.95
    max_new_tokens: 32768
EOF
}

# ─── main ─────────────────────────────────────────────────────────────────────

mkdir -p "${OUT_DIR}"
YAML_FILE="${OUT_DIR}/.model.yaml"

echo "=========================================="
echo "Model : ${MODEL_PATH}"
echo "Tasks : ${TASKS}"
echo "Output: ${OUT_DIR}"
echo "GPUs  : ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

# Repair tokenizer if model is a local directory
if [[ -d "${MODEL_PATH}" ]]; then
  fix_tokenizer_class_if_needed "${MODEL_PATH}"
fi

write_model_yaml "${MODEL_PATH}" "${YAML_FILE}"

lighteval vllm \
  "${YAML_FILE}" \
  "${TASKS}" \
  --custom-tasks "${CUSTOM_TASKS}" \
  --output-dir "${OUT_DIR}" \
  --save-details

echo "=========================================="
echo "Done: ${OUT_DIR}"
echo "=========================================="
