"""Check IDs/prompts, convert predictions as LLaVA does, and cross-check GQA accuracy."""
import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize(text):
    # Exact transformation in LLaVA scripts/convert_gqa_for_eval.py.
    return text.rstrip(".").lower()


def score(questions_path, annotations_path, predictions_path, output):
    questions = read_jsonl(questions_path)
    annotations = json.loads(annotations_path.read_text())
    answers = read_jsonl(predictions_path)
    expected_ids = [q["question_id"] for q in questions]
    ids = [a["question_id"] for a in answers]
    assert len(expected_ids) == len(set(expected_ids)), "Repeated input IDs"
    assert len(ids) == len(set(ids)), "Repeated prediction IDs"
    assert ids == expected_ids, "Missing, extra, or reordered predictions"
    assert set(ids) == set(annotations), "Annotation/prediction IDs differ"
    output.mkdir(parents=True, exist_ok=True)
    converted, scores, rows = [], [], []
    by_type = defaultdict(list)
    empty = 0
    for question, answer in zip(questions, answers):
        qid = question["question_id"]
        assert answer["prompt"] == question["text"], f"Prompt mismatch: {qid}"
        assert annotations[qid]["isBalanced"] is True
        raw = answer["text"]
        assert isinstance(raw, str)
        empty += int(not raw.strip())
        predicted = normalize(raw)
        gold = annotations[qid]["answer"]
        correct = int(predicted == gold)
        scores.append(correct)
        structural = annotations[qid]["types"]["structural"]
        semantic = annotations[qid]["types"]["semantic"]
        by_type[structural].append(correct)
        converted.append({"questionId": qid, "prediction": predicted})
        rows.append({"question_id": qid, "image_id": annotations[qid]["imageId"],
                     "question": annotations[qid]["question"], "reference": gold,
                     "prediction_raw": raw, "prediction_normalized": predicted,
                     "correct": correct, "structural_type": structural, "semantic_type": semantic})
    assert scores, "No predictions"
    converted_path = (output / "predictions-for-official.json").resolve()
    converted_path.write_text(json.dumps(converted, ensure_ascii=False))
    accuracy = 100 * sum(scores) / len(scores)
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("official_gqa_eval.py")),
        "--tier", "testdev_balanced", "--questions", str(annotations_path.resolve()),
        "--predictions", str(converted_path)], text=True, capture_output=True)
    (output / "official-score.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Official evaluator failed; inspect {output / 'official-score.log'}\n" + result.stderr[-2000:])
    match = re.search(r"^Accuracy: ([0-9.]+)%", result.stdout, re.MULTILINE)
    assert match and abs(float(match.group(1)) - accuracy) <= 0.0051, "Official accuracy cross-check failed"
    with (output / "per_question.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"samples": len(scores), "correct": sum(scores), "accuracy_percent": accuracy,
               "empty_predictions": empty, "official_accuracy_crosschecked": True,
               "accuracy_by_structural_type": {k: {"samples": len(v), "accuracy_percent": 100 * sum(v) / len(v)} for k, v in by_type.items()}}
    (output / "score.json").write_text(json.dumps(summary, indent=2))
    print(f"Verified {len(scores)} aligned predictions; empty predictions={empty}", flush=True)
    print(f"GQA accuracy: {accuracy:.4f}% ({sum(scores)}/{len(scores)}); official evaluator agrees", flush=True)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    score(a.questions, a.annotations, a.predictions, a.output)
