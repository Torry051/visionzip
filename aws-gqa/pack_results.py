"""Package only the completed latest full GQA run; never include model/data caches."""
import json
import tarfile
from pathlib import Path
import shutil


def main():
    project = Path.home() / "11785-visionzip"
    run = Path((project / "logs/latest-gqa-full-run.txt").read_text().strip()).resolve()
    assert run.parent == (project / "outputs").resolve() and run.name.startswith("gqa-full-")
    assert (run / "COMPLETE").is_file(), "Full run has not completed"
    assert json.loads((run / "summary.json").read_text())["samples"] == 12578
    questions = {}
    for line in (run / "questions-used.jsonl").read_text().splitlines():
        question = json.loads(line)
        questions[question["question_id"]] = question
    cases = json.loads((run / "example_cases.json").read_text())
    images = run / "case_images"
    images.mkdir(exist_ok=True)
    for group in cases.values():
        for case in group:
            filename = questions[case["question_id"]]["image"]
            shutil.copy2(project / "data/gqa/images" / filename, images / filename)
    output = project / "gqa-full-results.tar.gz"
    with tarfile.open(output, "w:gz") as archive:
        archive.add(run, arcname="outputs/" + run.name)
    print("Saved:", output)
    print("Bytes:", output.stat().st_size)


if __name__ == "__main__":
    main()
