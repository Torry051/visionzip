"""Prepare the pinned GQA testdev_balanced mirror for LLaVA's existing loader."""
import argparse
import hashlib
import io
import json
import re
import shutil
from pathlib import Path

DATASET = "lmms-lab-encoder/GQA"
REVISION = "a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8"
SUFFIX = "\nAnswer the question using a single word or phrase."


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def write_jsonl(path, rows):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.home() / "11785-visionzip/data/gqa")
    parser.add_argument("--parquet-dir", type=Path, help="Use already downloaded files for offline validation")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "metadata.json"
    if marker.exists():
        meta = json.loads(marker.read_text())
        assert meta["dataset_revision"] == REVISION, "Dataset revision changed"
        for name, expected in meta["sha256"].items():
            assert sha(root / name) == expected, f"Checksum mismatch: {name}"
        for name, expected in meta["image_sha256"].items():
            assert sha(root / "images" / name) == expected, f"Image checksum mismatch: {name}"
        print("Reused verified GQA data: 12578 questions / 398 images", flush=True)
        return
    if shutil.disk_usage(root).free < 2 * 1024**3:
        raise RuntimeError("Less than 2 GiB free; inspect disk space first.")

    import pyarrow.parquet as pq
    from PIL import Image

    paths = {}
    for kind in ["instructions", "images"]:
        config = "testdev_balanced_" + kind
        if args.parquet_dir:
            path = args.parquet_dir / (config + ".parquet")
        else:
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(DATASET,
                config + "/testdev-00000-of-00001.parquet",
                repo_type="dataset", revision=REVISION))
        paths[kind] = path
    rows = pq.read_table(paths["instructions"]).to_pylist()
    image_rows = pq.read_table(paths["images"]).to_pylist()
    assert len(rows) == 12578 and len(image_rows) == 398, "Unexpected split size"

    images = root / "images"
    images.mkdir(exist_ok=True)
    image_names, image_hashes = {}, {}
    for row in image_rows:
        image_id = row["id"]
        assert re.fullmatch(r"[A-Za-z0-9_-]+", image_id)
        assert image_id not in image_names, f"Duplicate image: {image_id}"
        raw = row["image"]["bytes"]
        assert isinstance(raw, bytes) and raw
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
            ext = {"JPEG": ".jpg", "PNG": ".png"}.get(image.format)
        assert ext, "Unexpected image encoding"
        filename = image_id + ext
        path = images / filename
        if path.exists():
            assert path.read_bytes() == raw, f"Existing image differs: {path}"
        else:
            path.write_bytes(raw)
        image_names[image_id] = filename
        image_hashes[filename] = sha(path)
    questions, annotations = [], {}
    for row in rows:
        qid = row["id"]
        assert isinstance(qid, str) and qid not in annotations, f"Duplicate question: {qid}"
        assert row["isBalanced"] is True
        assert isinstance(row["answer"], str) and row["answer"]
        assert isinstance(row["semantic"], list) and isinstance(row["types"], dict)
        assert row["imageId"] in image_names
        assert "\n" not in row["question"] and "\r" not in row["question"], "Unexpected question newline"
        questions.append({"question_id": qid, "image": image_names[row["imageId"]],
                          "text": row["question"] + SUFFIX})
        # Keep the source fields required by LLaVA's linked GQA evaluator.
        annotations[qid] = row
    assert len({a["imageId"] for a in annotations.values()}) == 398
    write_jsonl(root / "questions.jsonl", questions)
    write_json(root / "annotations.json", annotations)
    write_jsonl(root / "smoke_questions.jsonl", questions[:20])
    write_json(root / "smoke_annotations.json", {q["question_id"]: annotations[q["question_id"]] for q in questions[:20]})
    names = ["questions.jsonl", "annotations.json", "smoke_questions.jsonl", "smoke_annotations.json"]
    write_json(marker, {
        "dataset": DATASET, "dataset_revision": REVISION,
        "split": "testdev_balanced", "questions": len(questions), "unique_images": 398,
        "prompt": "Original question plus LLaVA short-answer instruction; no supplied OCR",
        "images": "Original encoded bytes from pinned mirror, no re-encoding",
        "source_sha256": {k: sha(p) for k, p in paths.items()},
        "sha256": {n: sha(root / n) for n in names}, "image_sha256": image_hashes,
        "scope": "Full GQA testdev_balanced mirror; not a claim of bit-for-bit original-paper protocol reproduction"
    })
    print("GQA DATA CHECK PASSED: 12578 questions / 398 images; smoke subset = 20 questions", flush=True)


if __name__ == "__main__":
    main()
