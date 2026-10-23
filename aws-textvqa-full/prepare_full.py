"""Prepare all TextVQA validation questions using the already-tested supplied-OCR prompt."""
import hashlib
import json
import shutil
from pathlib import Path

from datasets import load_dataset
from PIL import Image


def main():
    root = Path.home() / "11785-visionzip/data/textvqa"
    source = json.loads((root / "source.json").read_text())
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 10 * 1024**3:
        raise RuntimeError("Less than 10 GiB free; inspect disk usage before preparing full data.")
    ready_file = root / "full_metadata.json"
    if ready_file.exists():
        ready = json.loads(ready_file.read_text())
        valid = ready.get("revision") == source["revision"]
        for name in ["full_questions.jsonl", "full_annotations.json"]:
            path = root / name
            valid = valid and path.is_file()
            if path.is_file():
                valid = valid and hashlib.sha256(path.read_bytes()).hexdigest() == ready["sha256"][name]
        if valid:
            qs = [json.loads(line) for line in (root / "full_questions.jsonl").read_text().splitlines()]
            valid = len(qs) == 5000 and all((images / q["image"]).is_file() for q in qs)
        if valid:
            print("Previously prepared full dataset verified: 5000 questions. Reusing it.", flush=True)
            return
        raise RuntimeError("Existing full dataset metadata failed verification. Inspect it before rerunning.")
    print("Dataset:", source["dataset"], "revision:", source["revision"], flush=True)
    ds = load_dataset(source["dataset"], split="validation", revision=source["revision"], streaming=True)
    questions, annotations, question_ids, image_ids = [], [], set(), set()
    for row in ds:
        image_id = str(row["image_id"])
        qid = int(row["question_id"])
        if qid in question_ids:
            raise ValueError(f"Duplicate original question ID: {qid}")
        question_ids.add(qid)
        question = " ".join(row["question"].split())
        answers = row["answers"]
        if len(answers) != 10:
            raise ValueError(f"Expected 10 answers for {qid}")
        ocr = row["ocr_tokens"]
        if not isinstance(ocr, list) or not all(isinstance(t, str) for t in ocr):
            raise ValueError(f"Unexpected OCR for {qid}")
        ocr = [" ".join(t.split()) for t in ocr]
        image_name = image_id + ".png"
        if image_id not in image_ids:
            image_path = images / image_name
            if not image_path.exists():
                row["image"].convert("RGB").save(image_path)
            with Image.open(image_path) as im:
                im.verify()
            image_ids.add(image_id)
        prompt = question + "\nReference OCR token: " + ", ".join(ocr) + "\nAnswer the question using a single word or phrase."
        questions.append({"question_id": image_id, "image": image_name, "text": prompt, "textvqa_question_id": qid})
        annotations.append({"image_id": image_id, "question_id": qid, "question": question, "answers": answers, "ocr_tokens": ocr})
        if len(questions) % 100 == 0:
            print(f"Prepared {len(questions)} questions / {len(image_ids)} unique images", flush=True)
    if len(questions) != 5000 or len(image_ids) != 3166:
        raise ValueError(f"Unexpected split size: {len(questions)} questions, {len(image_ids)} images; inspect before running.")
    (root / "full_questions.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions))
    (root / "full_annotations.json").write_text(json.dumps({"data": annotations}, ensure_ascii=False))
    metadata = {**source, "questions": len(questions), "unique_images": len(image_ids),
                "selection": "All validation questions; multiple questions per image retained",
                "ocr_setting": "Supplied OCR; project prompt template, not yet checked against original LLaVA eval.zip",
                "normalization": "Whitespace within questions/OCR tokens collapsed; images saved as RGB PNG",
                "sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                           for name in ["full_questions.jsonl", "full_annotations.json"]}}
    ready_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("FULL DATA CHECK PASSED: 5000 questions, 3166 images, 10 reference answers each.", flush=True)


if __name__ == "__main__":
    main()
