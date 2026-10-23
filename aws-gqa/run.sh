#!/usr/bin/env bash
set -euo pipefail
KIT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$HOME/11785-visionzip"
RUN_KIND="${1:-smoke}"
if [[ "$RUN_KIND" != smoke && "$RUN_KIND" != full ]]; then
    echo 'Usage: bash aws-gqa/run.sh smoke|full' >&2
    exit 2
fi
source "$PROJECT_ROOT/activate-project.sh"
mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/outputs"
exec 9>"$PROJECT_ROOT/.gqa-eval.lock"
flock -n 9 || { echo 'A GQA job is already running.' >&2; exit 1; }
exec 8>"$PROJECT_ROOT/.full-textvqa.lock"
flock -n 8 || { echo 'TextVQA is running; wait for it before using this GPU.' >&2; exit 1; }
RUN_DIR="$PROJECT_ROOT/outputs/gqa-$RUN_KIND-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > "$PROJECT_ROOT/logs/latest-gqa-$RUN_KIND-run.txt"
trap 'echo "FAILED at line $LINENO. Inspect logs in $RUN_DIR. EC2 is still running."' ERR
echo "Run directory: $RUN_DIR"
echo "Stage 1/5: environment and full testdev_balanced data check"
python - <<'PY'
import sys
import torch, transformers, pyarrow, huggingface_hub, visionzip, tqdm
from llava.model.builder import load_pretrained_model
assert sys.version_info[:2] == (3, 10), "Activate the working project Python 3.10 environment"
assert torch.cuda.is_available(), "No CUDA GPU detected"
print("Python:", sys.executable)
print("Torch:", torch.__version__, "Transformers:", transformers.__version__)
print("GPU:", torch.cuda.get_device_name(0))
PY
python -u "$KIT_DIR/prepare_gqa.py" 2>&1 | tee "$RUN_DIR/data-preparation.log"
python -m pip freeze > "$RUN_DIR/requirements.txt"
nvidia-smi > "$RUN_DIR/gpu-info-before-inference.txt"
git -C "$PROJECT_ROOT/repos/LLaVA" rev-parse HEAD > "$RUN_DIR/llava-commit.txt"
git -C "$PROJECT_ROOT/repos/VisionZip" rev-parse HEAD > "$RUN_DIR/visionzip-commit.txt"
cp -R "$KIT_DIR" "$RUN_DIR/toolkit"
cp "$PROJECT_ROOT/data/gqa/metadata.json" "$RUN_DIR/data-metadata.json"
QUESTIONS="$PROJECT_ROOT/data/gqa/questions.jsonl"
ANNOTATIONS="$PROJECT_ROOT/data/gqa/annotations.json"
if [[ "$RUN_KIND" == smoke ]]; then
    QUESTIONS="$PROJECT_ROOT/data/gqa/smoke_questions.jsonl"
    ANNOTATIONS="$PROJECT_ROOT/data/gqa/smoke_annotations.json"
fi
cp "$QUESTIONS" "$RUN_DIR/questions-used.jsonl"
cp "$ANNOTATIONS" "$RUN_DIR/annotations-used.json"
python - "$RUN_DIR" "$RUN_KIND" <<'PY'
import json, sys
from pathlib import Path
config = {"run_kind": sys.argv[2], "dataset_split": "GQA testdev_balanced",
    "model": "liuhaotian/llava-v1.5-7b", "precision": "float16", "batch_size": 1,
    "temperature": 0, "num_beams": 1, "max_new_tokens": 128, "conv_mode": "vicuna_v1",
    "visionzip": {"dominant": 54, "contextual": 10},
    "prompt": "Original question + newline + Answer the question using a single word or phrase.",
    "ocr": "Not supplied", "scoring": "LLaVA conversion + linked GQA evaluator; binary exact accuracy",
    "timing": "Single inference-process wall times include loading; exploratory, not controlled latency or TTFT"}
(Path(sys.argv[1]) / "run-config.json").write_text(json.dumps(config, indent=2))
PY
COMMON_ARGS=(--model-path liuhaotian/llava-v1.5-7b --question-file "$QUESTIONS"
    --image-folder "$PROJECT_ROOT/data/gqa/images" --temperature 0 --num_beams 1
    --conv-mode vicuna_v1 --max_new_tokens 128)
for METHOD in baseline visionzip64; do
    echo "Stage: $METHOD inference ($RUN_KIND)"
    date -u
    START_SECONDS=$SECONDS
    python -u "$KIT_DIR/inference.py" --mode "$METHOD" "${COMMON_ARGS[@]}" \
        --answers-file "$RUN_DIR/$METHOD.jsonl" 2>&1 | tee "$RUN_DIR/$METHOD-inference.log"
    DURATION_SECONDS=$((SECONDS - START_SECONDS))
    printf '%s\t%s\n' "$METHOD" "$DURATION_SECONDS" >> "$RUN_DIR/inference-process-seconds.tsv"
    echo "Stage: validate and score $METHOD"
    python -u "$KIT_DIR/score_gqa.py" --questions "$QUESTIONS" --annotations "$ANNOTATIONS" \
        --predictions "$RUN_DIR/$METHOD.jsonl" --output "$RUN_DIR/$METHOD" \
        2>&1 | tee "$RUN_DIR/$METHOD-score.log"
done
echo 'Stage 5/5: paired analysis and token-count verification'
python -u "$KIT_DIR/analyze_gqa.py" --run-dir "$RUN_DIR" 2>&1 | tee "$RUN_DIR/analysis.log"
touch "$RUN_DIR/COMPLETE"
date -u
echo "ALL DONE ($RUN_KIND). Results: $RUN_DIR"
if [[ "$RUN_KIND" == smoke ]]; then
    echo 'This 20-question smoke run only checks the pipeline. Use full for a reportable full-split score.'
fi
echo 'EC2 has NOT been stopped automatically. Download results and stop it when finished.'
