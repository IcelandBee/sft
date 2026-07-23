#!/usr/bin/env python3
"""Build E5 by retaining all E4 T1 rows and a matched 20% local subset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile


class E5DatasetError(ValueError):
    """Raised when the frozen E4 source cannot produce the E5 protocol."""


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise E5DatasetError(f"invalid JSON at {path}:{number}") from exc
            if not isinstance(row, dict):
                raise E5DatasetError(f"non-object row at {path}:{number}")
            rows.append(row)
    return rows


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def decision(row: dict, location: str) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise E5DatasetError(f"{location} must contain three messages")
    try:
        payload = json.loads(messages[-1]["content"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise E5DatasetError(f"{location} has invalid assistant JSON") from exc
    value = payload.get("decision")
    if value not in {"GOOD", "BAD"}:
        raise E5DatasetError(f"{location} has invalid decision")
    return value


def build_e5_dataset(
    *,
    source_train: list[dict],
    source_manifest: list[dict],
    source_summary: dict,
    dev_bytes: bytes,
    output_dir: Path,
    local_pairs: int = 1247,
    seed: int = 42,
) -> dict:
    if output_dir.exists():
        raise E5DatasetError(f"output directory already exists: {output_dir}")
    if source_summary.get("output", {}).get("sample_type_counts") != {
        "T1_FULL": 9978,
        "T2_BAD": 3326,
        "T3_GOOD": 3326,
    }:
        raise E5DatasetError("unexpected E4 source sample counts")
    if source_summary.get("test_untouched") is not True:
        raise E5DatasetError("E4 source does not confirm Test isolation")
    if len(source_train) != 16630 or len(source_manifest) != 6652:
        raise E5DatasetError(
            f"unexpected E4 rows: train={len(source_train)} manifest={len(source_manifest)}"
        )

    local_by_index: dict[int, dict] = {}
    t2_by_candidate: dict[str, list[dict]] = defaultdict(list)
    t3_by_candidate: dict[str, list[dict]] = defaultdict(list)
    for row in source_manifest:
        index = row.get("train_output_index")
        sample_type = row.get("sample_type")
        if not isinstance(index, int) or not 0 <= index < len(source_train):
            raise E5DatasetError(f"invalid manifest train index: {index}")
        if index in local_by_index:
            raise E5DatasetError(f"duplicate manifest train index: {index}")
        local_by_index[index] = row
        if sample_type == "T2_BAD":
            candidate = row.get("candidate_id")
            if not isinstance(candidate, str):
                raise E5DatasetError("T2 row lacks candidate_id")
            t2_by_candidate[candidate].append(row)
        elif sample_type == "T3_GOOD":
            candidate = row.get("matched_t2_candidate_id")
            if not isinstance(candidate, str):
                raise E5DatasetError("T3 row lacks matched_t2_candidate_id")
            t3_by_candidate[candidate].append(row)
        else:
            raise E5DatasetError(f"unexpected manifest sample type: {sample_type}")

    candidate_ids = sorted(set(t2_by_candidate) & set(t3_by_candidate))
    if len(candidate_ids) < local_pairs:
        raise E5DatasetError(
            f"only {len(candidate_ids)} matched unique candidates for {local_pairs} pairs"
        )
    random.Random(seed).shuffle(candidate_ids)
    selected_candidates = candidate_ids[:local_pairs]

    t1_rows = [
        row for index, row in enumerate(source_train) if index not in local_by_index
    ]
    if len(t1_rows) != 9978:
        raise E5DatasetError(f"unexpected T1 rows: {len(t1_rows)}")

    selected_t2: list[dict] = []
    selected_t3: list[dict] = []
    category_counts: Counter[str] = Counter()
    for candidate in selected_candidates:
        t2_manifest = min(
            t2_by_candidate[candidate],
            key=lambda row: (row.get("candidate_occurrence", 999999), row["train_output_index"]),
        )
        t3_manifest = min(
            t3_by_candidate[candidate],
            key=lambda row: row["train_output_index"],
        )
        t2_row = source_train[t2_manifest["train_output_index"]]
        t3_row = source_train[t3_manifest["train_output_index"]]
        if decision(t2_row, f"T2 {candidate}") != "BAD":
            raise E5DatasetError(f"T2 decision mismatch: {candidate}")
        if decision(t3_row, f"T3 {candidate}") != "GOOD":
            raise E5DatasetError(f"T3 decision mismatch: {candidate}")
        selected_t2.append(t2_row)
        selected_t3.append(t3_row)
        category_counts[t2_manifest["category"]] += 1

    entries = [
        *[("T1_FULL", row) for row in t1_rows],
        *[("T2_BAD", row) for row in selected_t2],
        *[("T3_GOOD", row) for row in selected_t3],
    ]
    random.Random(seed).shuffle(entries)
    output_rows = [row for _, row in entries]
    sample_counts = Counter(sample_type for sample_type, _ in entries)
    expected_counts = {
        "T1_FULL": 9978,
        "T2_BAD": local_pairs,
        "T3_GOOD": local_pairs,
    }
    if sample_counts != expected_counts:
        raise E5DatasetError(f"unexpected E5 counts: {dict(sample_counts)}")

    image_count_distribution = Counter(len(row.get("images", [])) for row in output_rows)
    if image_count_distribution != {1: 9978, 2: local_pairs * 2}:
        raise E5DatasetError(
            f"unexpected image count distribution: {dict(image_count_distribution)}"
        )
    missing = [
        image
        for image in {image for row in output_rows for image in row["images"]}
        if not Path(image).is_file()
    ]
    if missing:
        raise E5DatasetError(f"missing image files: {len(missing)}; first={missing[0]}")

    train_bytes = jsonl_bytes(output_rows)
    decisions = Counter(
        decision(row, f"output row {index}") for index, row in enumerate(output_rows, 1)
    )
    summary = {
        "protocol_version": "e5_reduced_local_dataset_v1",
        "seed": seed,
        "source_protocol": source_summary.get("protocol_version"),
        "source_train_sha256": source_summary["output"]["train_sha256"],
        "test_untouched": True,
        "sample_type_counts": dict(sorted(sample_counts.items())),
        "decision_counts": dict(sorted(decisions.items())),
        "unique_t2_candidates": len(selected_candidates),
        "unique_t3_good_sources": len(
            {row["images"][0] for row in selected_t3}
        ),
        "t2_category_counts": dict(sorted(category_counts.items())),
        "train_rows": len(output_rows),
        "train_sha256": sha256_bytes(train_bytes),
        "dev_sha256": sha256_bytes(dev_bytes),
        "mix": "T1:local=80:20; local_BAD:local_GOOD=1:1",
        "status": "PASS",
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        (staging / "train.jsonl").write_bytes(train_bytes)
        (staging / "dev.jsonl").write_bytes(dev_bytes)
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
    parser.add_argument("--source-data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--local-pairs", type=int, default=1247)
    args = parser.parse_args()
    try:
        train_path = args.source_data / "train.jsonl"
        manifest_path = args.source_data / "local_manifest.jsonl"
        summary_path = args.source_data / "build_summary.json"
        dev_path = args.source_data / "dev.jsonl"
        summary = build_e5_dataset(
            source_train=load_jsonl(train_path),
            source_manifest=load_jsonl(manifest_path),
            source_summary=json.loads(summary_path.read_text(encoding="utf-8")),
            dev_bytes=dev_path.read_bytes(),
            output_dir=args.output_dir,
            local_pairs=args.local_pairs,
        )
    except (E5DatasetError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print("=== E5 REDUCED-LOCAL DATASET ===")
    print(f"sample_types={summary['sample_type_counts']}")
    print(f"decisions={summary['decision_counts']}")
    print(f"unique_t2={summary['unique_t2_candidates']}")
    print(f"unique_t3_good={summary['unique_t3_good_sources']}")
    print(f"train_sha256={summary['train_sha256']}")
    print(f"output={args.output_dir}")
    print("E5_DATASET_BUILD: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
