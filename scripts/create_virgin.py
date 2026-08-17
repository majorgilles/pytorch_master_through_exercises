"""Create a pristine retry copy of a progressive exercise notebook."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

ANSWER_ID = re.compile(r"^exercise-\d{3}-answer$")
WORKING_NOTE = (
    "> **Working copy:** This file may contain learner answers and scratch work. "
    "Use the paired `_virgin.ipynb` notebook whenever you want a clean retry."
)
VIRGIN_NOTE = (
    "> **Virgin retry copy:** Answer cells are intentionally blank and all saved "
    "execution outputs are cleared. Copy this file before a fresh attempt; do not "
    "record ongoing work in the virgin file."
)


def _source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def _leading_scaffold_comments(source: str) -> str:
    """Keep only comments and blank lines before the first executable line."""
    kept: list[str] = []
    for line in source.splitlines(keepends=True):
        if not line.strip() or line.lstrip().startswith("#"):
            kept.append(line)
            continue
        break
    return "".join(kept).rstrip() + "\n"


def create_virgin(source_path: Path, destination_path: Path) -> None:
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    answer_count = 0

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "markdown":
            text = _source_text(cell)
            text = text.replace(WORKING_NOTE, VIRGIN_NOTE)
            text = text.replace(
                "This working copy may contain learner answers;",
                "This virgin copy contains no learner answers;",
            )
            text = text.replace(
                "This working copy may contain learner solutions.",
                "This virgin copy contains no learner solutions.",
            )
            text = text.replace(
                "This working copy may contain learner solutions;",
                "This virgin copy contains no learner solutions;",
            )
            cell["source"] = text.splitlines(keepends=True)
            continue

        if cell.get("cell_type") != "code":
            continue

        cell["execution_count"] = None
        cell["outputs"] = []

        if ANSWER_ID.fullmatch(cell.get("id", "")):
            cell["source"] = _leading_scaffold_comments(_source_text(cell)).splitlines(
                keepends=True
            )
            answer_count += 1

    if answer_count == 0:
        raise ValueError(
            "No answer cells were found. Expected IDs such as `exercise-001-answer`."
        )

    # Refuse to write a supposed virgin notebook containing executable answer code.
    for cell in notebook["cells"]:
        if not ANSWER_ID.fullmatch(cell.get("id", "")):
            continue
        tree = ast.parse(_source_text(cell))
        if tree.body:
            raise ValueError(f"Answer cell {cell['id']} still contains executable code.")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Created {destination_path} with {answer_count} blank answer cells.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blank structured answer cells and clear all notebook outputs."
    )
    parser.add_argument("source", type=Path, help="Working `*_exercises.ipynb` file")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination; defaults to `<stem>_virgin.ipynb` beside the source",
    )
    args = parser.parse_args()

    source_path = args.source.resolve()
    if source_path.stem.endswith("_virgin"):
        parser.error("Pass the working notebook, not an existing `_virgin` copy.")
    if not source_path.is_file():
        parser.error(f"Notebook not found: {source_path}")

    destination = args.output
    if destination is None:
        destination = source_path.with_name(f"{source_path.stem}_virgin.ipynb")
    create_virgin(source_path, destination.resolve())


if __name__ == "__main__":
    main()
