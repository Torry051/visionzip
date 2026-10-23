import argparse
import json
from pathlib import Path


def check(question_file, prediction_file):
    questions = [json.loads(line) for line in Path(question_file).read_text().splitlines() if line.strip()]
    results = [json.loads(line) for line in Path(prediction_file).read_text().splitlines() if line.strip()]
    if len(questions) != 5000 or len(results) != len(questions):
        raise ValueError(f"Expected 5000 predictions; questions={len(questions)}, predictions={len(results)}")
    for i, (q, r) in enumerate(zip(questions, results)):
        if r["question_id"] != q["question_id"] or r["prompt"] != q["text"]:
            raise ValueError(f"Sample identity/prompt mismatch at row {i + 1}")
        if not isinstance(r["text"], str):
            raise ValueError(f"Invalid prediction at row {i + 1}")
    empty = sum(not r["text"].strip() for r in results)
    print(f"Verified 5000 aligned predictions; empty predictions={empty} (kept for scoring).")
    return questions, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("questions")
    parser.add_argument("predictions")
    args = parser.parse_args()
    check(args.questions, args.predictions)
