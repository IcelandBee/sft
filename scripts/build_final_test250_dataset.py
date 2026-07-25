#!/usr/bin/env python3
"""Build the frozen 250-image final Test JSONL from the reviewed workbook."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

try:
    from scripts.audit_final_test250_independence import (
        TestIndependenceError,
        classify_label,
        load_sheet_rows,
    )
    from scripts.evaluate_e1_dev import SYSTEM_PROMPT, USER_PROMPT
    from scripts.inspect_final_test250_sources import IMAGE_SUFFIXES, file_sha256
except ModuleNotFoundError:  # Support absolute-path execution on the server.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.audit_final_test250_independence import (
        TestIndependenceError,
        classify_label,
        load_sheet_rows,
    )
    from scripts.evaluate_e1_dev import SYSTEM_PROMPT, USER_PROMPT
    from scripts.inspect_final_test250_sources import IMAGE_SUFFIXES, file_sha256


class FinalTestBuildError(ValueError):
    """Raised when the frozen Test250 contract cannot be built exactly."""


def jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def build_dataset(
    *,
    workbook_path: Path,
    image_root: Path,
    output_dir: Path,
    expected_workbook_sha256: str,
) -> dict:
    if output_dir.exists():
        raise FinalTestBuildError(f"output directory already exists: {output_dir}")
    workbook_sha256 = file_sha256(workbook_path)
    if workbook_sha256 != expected_workbook_sha256:
        raise FinalTestBuildError(
            f"workbook sha256 mismatch: {workbook_sha256} != {expected_workbook_sha256}"
        )
    rows = load_sheet_rows(workbook_path)
    if len(rows) != 251:
        raise FinalTestBuildError(f"expected header + 250 rows, got {len(rows)}")
    header = rows[0]
    if header.get(6) != "image_version" or header.get(7) != "group_index":
        raise FinalTestBuildError("unexpected workbook schema")

    images = sorted(
        (
            path.resolve()
            for path in image_root.rglob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )
    by_name = {path.name.casefold(): path for path in images}
    output_rows: list[dict] = []
    manifest_rows: list[dict] = []
    labels: Counter[str] = Counter()
    selected_paths: set[str] = set()
    selected_digest = hashlib.sha256()
    for excel_row, row in enumerate(rows[1:], start=2):
        group = str(row.get(7, "")).strip()
        version = str(row.get(6, "")).strip()
        label = classify_label(row.get(9))
        target_prefix = f"{group}_{version}_".casefold()
        matches = sorted(name for name in by_name if name.startswith(target_prefix))
        if len(matches) != 1:
            raise FinalTestBuildError(
                f"expected one uploaded image at Excel row {excel_row}, got {matches}"
            )
        image = by_name[matches[0]]
        if str(image) in selected_paths:
            raise FinalTestBuildError(f"duplicate target image: {image}")
        selected_paths.add(str(image))
        labels[label] += 1
        if label == "GOOD":
            gold = {"decision": "GOOD", "categories": [], "reasons": []}
        else:
            # The workbook contains a binary label only. These auxiliary fields are
            # protocol placeholders and are never used in the confusion matrix.
            gold = {
                "decision": "BAD",
                "categories": ["其他"],
                "reasons": ["人工标注为有异常"],
            }
        output_rows.append(
            {
                "images": [str(image)],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT},
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            gold, ensure_ascii=False, separators=(",", ":")
                        ),
                    },
                ],
            }
        )
        image_sha256 = file_sha256(image)
        selected_digest.update(
            (
                f"{excel_row}\0{group}\0{label}\0{image.name}\0"
                f"{image.stat().st_size}\0{image_sha256}\n"
            ).encode("utf-8")
        )
        manifest_rows.append(
            {
                "test_index": len(output_rows) - 1,
                "excel_row": excel_row,
                "source_id": row.get(1),
                "group_index": group,
                "image_version": version,
                "image_path": str(image),
                "image_sha256": image_sha256,
                "gold_decision": label,
                "auxiliary_gold_is_placeholder": label == "BAD",
            }
        )

    if labels != {"GOOD": 186, "BAD": 64}:
        raise FinalTestBuildError(f"unexpected labels: {dict(labels)}")
    test_bytes = jsonl_bytes(output_rows)
    manifest_bytes = jsonl_bytes(manifest_rows)
    summary = {
        "protocol_version": "final_test250_dataset_v1",
        "workbook": str(workbook_path.resolve()),
        "workbook_sha256": workbook_sha256,
        "image_root": str(image_root.resolve()),
        "rows": len(output_rows),
        "label_counts": dict(sorted(labels.items())),
        "selected_image_manifest_sha256": selected_digest.hexdigest(),
        "test_jsonl_sha256": hashlib.sha256(test_bytes).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "prompt_protocol": "e1_dev_generation_v1",
        "bad_auxiliary_fields": "generic placeholders; binary decision is authoritative",
        "checkpoint_selection_forbidden": True,
        "prompt_selection_forbidden": True,
        "status": "PASS",
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        (staging / "test.jsonl").write_bytes(test_bytes)
        (staging / "manifest.jsonl").write_bytes(manifest_bytes)
        (staging / "build_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-workbook-sha256", required=True)
    args = parser.parse_args()
    try:
        summary = build_dataset(
            workbook_path=args.workbook,
            image_root=args.image_root,
            output_dir=args.output_dir,
            expected_workbook_sha256=args.expected_workbook_sha256,
        )
    except (FinalTestBuildError, TestIndependenceError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("=== FINAL TEST250 DATASET ===")
    print(f"rows={summary['rows']} labels={summary['label_counts']}")
    print(f"test_jsonl_sha256={summary['test_jsonl_sha256']}")
    print(f"selected_image_manifest_sha256={summary['selected_image_manifest_sha256']}")
    print(f"output={args.output_dir}")
    print("FINAL_TEST250_DATASET_BUILD: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
