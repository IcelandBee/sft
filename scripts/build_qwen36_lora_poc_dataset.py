#!/usr/bin/env python3
"""Build an interleaved 20-row E5 dataset for the Qwen3.6 LoRA smoke run."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

try:
    from scripts.check_qwen36_processor_preflight import (
        ProcessorPreflightError,
        load_jsonl,
        select_poc_rows,
        validate_e5_distribution,
    )
except ModuleNotFoundError:  # Support direct execution via an absolute script path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.check_qwen36_processor_preflight import (
        ProcessorPreflightError,
        load_jsonl,
        select_poc_rows,
        validate_e5_distribution,
    )


STRATA_ORDER = ("T1_GOOD", "T1_BAD", "T2_BAD", "T3_GOOD")


def build_interleaved_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    selected = select_poc_rows(rows)
    grouped: dict[str, list[tuple[int, dict]]] = {name: [] for name in STRATA_ORDER}
    for source_index, stratum, row in selected:
        grouped[stratum].append((source_index, row))
    if any(len(grouped[name]) != 5 for name in STRATA_ORDER):
        raise ProcessorPreflightError(
            f"unexpected selected counts: {dict((k, len(v)) for k, v in grouped.items())}"
        )
    poc_rows: list[dict] = []
    manifest_rows: list[dict] = []
    for round_index in range(5):
        for stratum in STRATA_ORDER:
            source_index, row = grouped[stratum][round_index]
            poc_rows.append(row)
            manifest_rows.append(
                {
                    "poc_index": len(poc_rows) - 1,
                    "source_index": source_index,
                    "stratum": stratum,
                    "image_count": len(row["images"]),
                }
            )
    return poc_rows, manifest_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() or args.manifest.exists():
        print("ERROR: output or manifest already exists", file=sys.stderr)
        return 2
    try:
        source = args.train.read_bytes()
        rows = load_jsonl(args.train)
        distribution = validate_e5_distribution(rows)
        poc_rows, manifest_rows = build_interleaved_rows(rows)
        missing = [
            image
            for row in poc_rows
            for image in row["images"]
            if not Path(image).is_file()
        ]
        if missing:
            raise ProcessorPreflightError(
                f"missing selected images: {len(missing)}; first={missing[0]}"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        text = "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in poc_rows
        )
        args.output.write_text(text, encoding="utf-8")
        manifest = {
            "protocol_version": "qwen36_lora_poc20_dataset_v1",
            "source": str(args.train),
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "source_distribution": dict(distribution),
            "rows": len(poc_rows),
            "image_inputs": sum(len(row["images"]) for row in poc_rows),
            "selected_distribution": dict(
                Counter(row["stratum"] for row in manifest_rows)
            ),
            "interleave_order": list(STRATA_ORDER),
            "poc_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "samples": manifest_rows,
            "dev_untouched": True,
            "test_untouched": True,
        }
        args.manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, ProcessorPreflightError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"rows={manifest['rows']} image_inputs={manifest['image_inputs']} "
        f"distribution={manifest['selected_distribution']}"
    )
    print(f"poc={args.output}")
    print("QWEN36_LORA_POC_DATASET: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
