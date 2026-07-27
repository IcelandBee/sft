#!/usr/bin/env python3
"""Build a prediction-free review manifest for the 73 priority Test250 rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile


class PriorityReviewBuildError(ValueError):
    """Raised when the diagnostic source cannot produce the frozen review set."""


def _load_jsonl(path: Path) -> list[dict]:
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise PriorityReviewBuildError(f"cannot read priority source: {path}") from exc
    rows: list[dict] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PriorityReviewBuildError(
                f"invalid JSON at {path}:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise PriorityReviewBuildError(
                f"row at {path}:{line_number} must be an object"
            )
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    source_rows: list[dict], *, expected_count: int = 73, seed: int = 20260727
) -> list[dict]:
    """Strip all gold/model fields and return a reproducibly shuffled manifest."""
    if len(source_rows) != expected_count:
        raise PriorityReviewBuildError(
            f"expected {expected_count} priority rows, got {len(source_rows)}"
        )
    safe_rows: list[dict] = []
    seen_rows: set[int] = set()
    seen_images: set[str] = set()
    for source in source_rows:
        row = source.get("row")
        index = source.get("index")
        image_path = source.get("image_path")
        if not isinstance(row, int) or row < 1 or row in seen_rows:
            raise PriorityReviewBuildError(f"invalid or duplicate Test row: {row}")
        if index != row - 1:
            raise PriorityReviewBuildError(f"Test index mismatch at row {row}")
        if (
            not isinstance(image_path, str)
            or not image_path
            or image_path in seen_images
        ):
            raise PriorityReviewBuildError(f"invalid or duplicate image at row {row}")
        seen_rows.add(row)
        seen_images.add(image_path)
        safe_rows.append({"row": row, "index": index, "image_path": image_path})

    random.Random(seed).shuffle(safe_rows)
    return [
        {"review_order": order, **record}
        for order, record in enumerate(safe_rows, start=1)
    ]


def run_build(
    source_path: Path,
    output_dir: Path,
    *,
    expected_count: int = 73,
    seed: int = 20260727,
) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise PriorityReviewBuildError(
            f"output directory already exists: {output_dir}"
        )
    manifest = build_manifest(
        _load_jsonl(source_path), expected_count=expected_count, seed=seed
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        review_text = "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in manifest
        )
        (staging / "review.jsonl").write_text(review_text, encoding="utf-8")
        summary = {
            "protocol_version": "final_test250_priority_review73_v1",
            "rows": len(manifest),
            "shuffle_seed": seed,
            "source_path": str(source_path),
            "source_sha256": _sha256(source_path),
            "review_manifest_sha256": hashlib.sha256(
                review_text.encode("utf-8")
            ).hexdigest(),
            "contains_original_labels": False,
            "contains_model_predictions": False,
            "selection_is_model_informed": True,
            "use_restriction": "rapid_diagnostic_review; not an independent final Test",
        }
        (staging / "build_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=73)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()
    summary = run_build(
        args.source,
        args.output_dir,
        expected_count=args.expected_count,
        seed=args.seed,
    )
    print("=== FINAL TEST250 PRIORITY REVIEW MANIFEST ===")
    print(f"rows={summary['rows']} shuffle_seed={summary['shuffle_seed']}")
    print(f"review_manifest_sha256={summary['review_manifest_sha256']}")
    print("contains_original_labels=False contains_model_predictions=False")
    print(f"output={args.output_dir}")
    print("FINAL_TEST250_PRIORITY_REVIEW_BUILD: PASS")


if __name__ == "__main__":
    main()
