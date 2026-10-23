#!/usr/bin/env bash
set -euo pipefail
KIT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$HOME/11785-visionzip"
source "$PROJECT_ROOT/activate-project.sh"
mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/outputs"
exec 9>"$PROJECT_ROOT/.full-textvqa.lock"
if ! flock -n 9; then
    echo 'Another full evaluation is running; refusing to launch a duplicate.' >&2
    exit 1
fi
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="$PROJECT_ROOT/outputs/textvqa-full-$RUN_ID"
mkdir "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > "$PROJECT_ROOT/logs/latest-full-run.txt"
trap 'echo "FAILED at line $LINENO. Results/logs remain in $RUN_DIR. EC2 is still running."' ERR
echo "Run directory: $RUN_DIR"
echo 'Step 1/6: prepare full validation data'
python -u "$KIT_DIR/prepare_full.py" 2>&1 | tee "$RUN_DIR/data-preparation.log"
python -m pip freeze > "$RUN_DIR/requirements.txt"
nvidia-smi > "$RUN_DIR/gpu-info.txt"
git -C "$PROJECT_ROOT/repos/LLaVA" rev-parse HEAD > "$RUN_DIR/llava-commit.txt"
git -C "$PROJECT_ROOT/repos/VisionZip" rev-parse HEAD > "$RUN_DIR/visionzip-commit.txt"
cp "$PROJECT_ROOT/data/textvqa/full_metadata.json" "$RUN_DIR/data-metadata.json"
cp "$PROJECT_ROOT/data/textvqa/full_questions.jsonl" "$RUN_DIR/questions-used.jsonl"
cp "$PROJECT_ROOT/data/textvqa/full_annotations.json" "$RUN_DIR/annotations-used.json"
cp "$PROJECT_ROOT/visionzip_eval.py" "$RUN_DIR/visionzip_eval-used.py"
python - "$RUN_DIR" <<'PY'
import json, sys
from pathlib import Path
config = {"model": "liuhaotian/llava-v1.5-7b", "batch_size": 1, "precision": "float16",
          "temperature": 0, "num_beams": 1, "conversation": "vicuna_v1", "max_new_tokens": 128,
          "modes": ["uncompressed", "VisionZip dominant=54 contextual=10"],
          "timing_note": "Log throughput is exploratory; this run is not a controlled TTFT/latency benchmark."}
(Path(sys.argv[1]) / "run-config.json").write_text(json.dumps(config, indent=2))
PY
QUESTIONS="$PROJECT_ROOT/data/textvqa/full_questions.jsonl"
ANNOTATIONS="$PROJECT_ROOT/data/textvqa/full_annotations.json"
COMMON_ARGS=(--model-path liuhaotian/llava-v1.5-7b --question-file "$QUESTIONS"
    --image-folder "$PROJECT_ROOT/data/textvqa/images" --temperature 0 --num_beams 1
    --conv-mode vicuna_v1 --max_new_tokens 128)
echo 'Step 2/6: uncompressed LLaVA inference'
date -u
python -u -m llava.eval.model_vqa_loader "${COMMON_ARGS[@]}" \
    --answers-file "$RUN_DIR/baseline.jsonl" 2>&1 | tee "$RUN_DIR/baseline-inference.log"
echo 'Step 3/6: validate and score baseline'
python "$KIT_DIR/check_predictions.py" "$QUESTIONS" "$RUN_DIR/baseline.jsonl"
python -m llava.eval.eval_textvqa --annotation-file "$ANNOTATIONS" \
    --result-file "$RUN_DIR/baseline.jsonl" 2>&1 | tee "$RUN_DIR/baseline-score.log"
echo 'Step 4/6: VisionZip-64 inference'
date -u
python -u "$PROJECT_ROOT/visionzip_eval.py" "${COMMON_ARGS[@]}" \
    --answers-file "$RUN_DIR/visionzip64.jsonl" 2>&1 | tee "$RUN_DIR/visionzip64-inference.log"
echo 'Step 5/6: validate and score VisionZip'
python "$KIT_DIR/check_predictions.py" "$QUESTIONS" "$RUN_DIR/visionzip64.jsonl"
python -m llava.eval.eval_textvqa --annotation-file "$ANNOTATIONS" \
    --result-file "$RUN_DIR/visionzip64.jsonl" 2>&1 | tee "$RUN_DIR/visionzip64-score.log"
echo 'Step 6/6: paired analysis and example cases'
python "$KIT_DIR/analyze_full.py" --run-dir "$RUN_DIR"
touch "$RUN_DIR/COMPLETE"
date -u
echo "ALL DONE. Results: $RUN_DIR"
echo 'EC2 has NOT been stopped automatically. Download results and stop it when finished.'
