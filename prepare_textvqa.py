import argparse
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, choices=[10, 100], default=10)
args = parser.parse_args()

root = Path.home() / "11785-visionzip/data/textvqa"
images_dir = root / "images"
images_dir.mkdir(parents=True, exist_ok=True)

repo = "lmms-lab-encoder/textvqa"
source_file = root / "source.json"


if source_file.exists():
    source = json.loads(source_file.read_text())
    assert source["dataset"] == repo
    revision = source["revision"]
else:
    revision = HfApi().dataset_info(repo).sha
    source = {
        "dataset": repo,
        "revision": revision,
        "split": "validation",
    }
    source_file.write_text(json.dumps(source, indent=2))

print("Dataset:", repo, flush=True)
print("Revision:", revision, flush=True)
print("Reading validation samples...", flush=True)

dataset = load_dataset(
    repo,
    split="validation",
    revision=revision,
    streaming=True,
)

questions = []
annotations = []
seen_images = set()

for row in dataset:
    image_id = str(row["image_id"])
    if image_id in seen_images:
        continue

    question = row["question"].strip()
    answers = row["answers"]
    ocr_tokens = row["ocr_tokens"]

    assert len(answers) == 10, "Expected 10 human answers"
    assert isinstance(ocr_tokens, list), "Unexpected OCR format"
    assert all(isinstance(t, str) for t in ocr_tokens)

    
    image = row["image"].convert("RGB")
    image_name = image_id + ".png"
    image.save(images_dir / image_name)

    
    prompt = (
        question
        + "\nReference OCR token: "
        + ", ".join(ocr_tokens)
        + "\nAnswer the question using a single word or phrase."
    )

    
    questions.append({
        "question_id": image_id,
        "image": image_name,
        "text": prompt,
        "textvqa_question_id": row["question_id"],
    })

    annotations.append({
        "image_id": image_id,
        "question_id": row["question_id"],
        "question": question,
        "answers": answers,
        "ocr_tokens": ocr_tokens,
    })

    seen_images.add(image_id)
    print(f"Prepared {len(questions)}/{args.n}", flush=True)

    if len(questions) == args.n:
        break

assert len(questions) == args.n, "Not enough samples"

question_file = root / f"pilot{args.n}_questions.jsonl"
annotation_file = root / f"pilot{args.n}_annotations.json"

with question_file.open("w") as f:
    for question in questions:
        f.write(json.dumps(question, ensure_ascii=False) + "\n")

annotation_file.write_text(
    json.dumps({"data": annotations}, ensure_ascii=False, indent=2)
)

metadata = {
    **source,
    "sample_count": len(questions),
    "unique_images": len(seen_images),
    "selection": "First N distinct images in validation stream, one question each",
    "ocr_setting": "Supplied OCR tokens included in prompt",
    "purpose": "Pipeline pilot; not full-split reproduction",
    "question_ids": [a["question_id"] for a in annotations],
}

(root / f"pilot{args.n}_metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2)
)

print("\nData preparation complete.")
print("Questions:", question_file)
print("Annotations:", annotation_file)
print("Images:", images_dir)
print("\nFirst prompt:\n" + questions[0]["text"])
