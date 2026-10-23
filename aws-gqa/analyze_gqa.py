"""Paired full-split comparison; image-cluster resampling respects repeated images."""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    a = p.parse_args()
    root = a.run_dir
    base = list(csv.DictReader((root / "baseline/per_question.csv").open()))
    compressed = list(csv.DictReader((root / "visionzip64/per_question.csv").open()))
    assert [r["question_id"] for r in base] == [r["question_id"] for r in compressed]
    n = len(base)
    for method, tokens in [("baseline", 576), ("visionzip64", 64)]:
        observed = re.findall(r"Projected visual tokens: (\d+)", (root / f"{method}-inference.log").read_text())
        assert len(observed) == n and set(observed) == {str(tokens)}, f"Token count validation failed: {method}"
    groups = defaultdict(lambda: [0, 0])
    paired = []
    for b, z in zip(base, compressed):
        assert b["image_id"] == z["image_id"] and b["reference"] == z["reference"]
        delta = int(z["correct"]) - int(b["correct"])
        groups[b["image_id"]][0] += delta
        groups[b["image_id"]][1] += 1
        paired.append({"question_id": b["question_id"], "image_id": b["image_id"],
                       "question": b["question"], "reference": b["reference"],
                       "baseline": b["prediction_raw"], "visionzip64": z["prediction_raw"],
                       "baseline_correct": int(b["correct"]), "visionzip64_correct": int(z["correct"]),
                       "difference": delta})
    grouped = np.array(list(groups.values()), dtype=float)
    rng = np.random.default_rng(785)
    boot = []
    for _ in range(2000):
        selected = grouped[rng.integers(0, len(grouped), len(grouped))]
        boot.append(100 * selected[:, 0].sum() / selected[:, 1].sum())
    base_score = 100 * sum(int(b["correct"]) for b in base) / n
    zip_score = 100 * sum(int(z["correct"]) for z in compressed) / n
    result = {"samples": n, "unique_images": len(groups),
              "baseline_accuracy_percent": base_score, "visionzip64_accuracy_percent": zip_score,
              "difference_percentage_points": zip_score - base_score,
              "paired_image_bootstrap_difference_95ci_pp": np.percentile(boot, [2.5, 97.5]).tolist(),
              "bootstrap_repetitions": 2000, "bootstrap_seed": 785,
              "improved_questions": sum(r["difference"] > 0 for r in paired),
              "worsened_questions": sum(r["difference"] < 0 for r in paired),
              "equal_questions": sum(r["difference"] == 0 for r in paired),
              "protocol": "GQA testdev_balanced pinned mirror; LLaVA short-answer prompt; greedy; FP16; batch 1",
              "scope": "Training-free baseline comparison; cross-layer extension not implemented. Single process times are exploratory."}
    with (root / "per_question.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    cases = {"improved": [r for r in paired if r["difference"] > 0][:5],
             "worsened": [r for r in paired if r["difference"] < 0][:5]}
    (root / "example_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    (root / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
