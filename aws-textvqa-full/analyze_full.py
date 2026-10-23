"""Use LLaVA's answer normalization/soft scores; bootstrap whole image clusters."""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from llava.eval.eval_textvqa import prompt_processor
from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator

from check_predictions import check


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    root = Path.home() / "11785-visionzip/data/textvqa"
    q, baseline = check(root / "full_questions.jsonl", args.run_dir / "baseline.jsonl")
    _, compressed = check(root / "full_questions.jsonl", args.run_dir / "visionzip64.jsonl")
    anns = json.loads((root / "full_annotations.json").read_text())["data"]
    # Same image + normalized-question lookup as the official eval_textvqa entry point.
    lookup = {(a["image_id"], a["question"].lower()): a for a in anns}
    evaluator = TextVQAAccuracyEvaluator()
    rows = []
    for question, b, z in zip(q, baseline, compressed):
        ann = lookup[(b["question_id"], prompt_processor(b["prompt"]))]
        answer_scores = evaluator._compute_answer_scores(ann["answers"])
        bs = answer_scores.get(evaluator.answer_processor(b["text"]), 0.0)
        zs = answer_scores.get(evaluator.answer_processor(z["text"]), 0.0)
        rows.append({"image_id": ann["image_id"], "question_id": question["textvqa_question_id"],
                     "image_file": str(root / "images" / question["image"]), "question": ann["question"],
                     "reference_answers": json.dumps(ann["answers"], ensure_ascii=False),
                     "baseline_prediction": b["text"], "visionzip64_prediction": z["text"],
                     "baseline_score": bs, "visionzip64_score": zs, "delta": zs - bs})
    with (args.run_dir / "per_question.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for row in rows:
        sums = groups.setdefault(row["image_id"], [0.0, 0])
        sums[0] += row["delta"]
        sums[1] += 1
    sums = np.array([v[0] for v in groups.values()])
    counts = np.array([v[1] for v in groups.values()])
    rng = np.random.default_rng(785)
    boot = []
    for _ in range(20):
        indices = rng.integers(0, len(groups), size=(100, len(groups)))
        boot.extend((sums[indices].sum(axis=1) / counts[indices].sum(axis=1)).tolist())
    low, high = np.quantile(boot, [0.025, 0.975])
    summary = {"samples": len(rows), "unique_images": len(groups),
               "baseline_accuracy_percent": 100 * float(np.mean([r["baseline_score"] for r in rows])),
               "visionzip64_accuracy_percent": 100 * float(np.mean([r["visionzip64_score"] for r in rows])),
               "difference_percentage_points": 100 * float(np.mean([r["delta"] for r in rows])),
               "paired_image_bootstrap_difference_95ci_pp": [100 * float(low), 100 * float(high)],
               "bootstrap_repetitions": 2000, "bootstrap_seed": 785,
               "improved_questions": sum(r["delta"] > 0 for r in rows),
               "worsened_questions": sum(r["delta"] < 0 for r in rows),
               "equal_questions": sum(r["delta"] == 0 for r in rows),
               "protocol_note": "Full validation coverage using the project's supplied-OCR prompt. Not yet matched to original-paper preprocessing.",
               "scope_note": "Comparison of existing baseline methods; cross-layer extension not yet implemented."}
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    cases = {"improvements": sorted([r for r in rows if r["delta"] > 0], key=lambda r: -r["delta"])[:10],
             "regressions": sorted([r for r in rows if r["delta"] < 0], key=lambda r: r["delta"])[:10],
             "both_zero": [r for r in rows if r["baseline_score"] == r["visionzip64_score"] == 0][:10]}
    case_images = args.run_dir / "case_images"
    case_images.mkdir(exist_ok=True)
    for category in cases.values():
        for row in category[:3]:
            destination = case_images / Path(row["image_file"]).name
            shutil.copy2(row["image_file"], destination)
            row["copied_image"] = str(Path("case_images") / destination.name)
    (args.run_dir / "example_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    print(json.dumps(summary, indent=2))
    print("Saved per_question.csv and example_cases.json. Cases are selected illustrations, not representative estimates.")


if __name__ == "__main__":
    main()
