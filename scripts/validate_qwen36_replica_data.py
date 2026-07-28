#!/usr/bin/env python3
"""Validate frozen Train/Dev contracts for Qwen3.6 E1-E5 replicas."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


DEV_SHA256 = "cd3a1e2d215b505526b7382a9ccf8d9acaca540e60dedf684cfaeca28cde3acb"
CONTRACTS = {
    "E1": {"rows": 9978, "images": {1: 9978}, "decisions": {"GOOD": 6074, "BAD": 3904}},
    "E2": {"rows": 9978, "images": {1: 9978}, "decisions": {"GOOD": 6074, "BAD": 3904}},
    "E3": {"rows": 9978, "images": {1: 9978}, "decisions": {"GOOD": 6074, "BAD": 3904}},
    "E4": {
        "rows": 16630,
        "images": {1: 9978, 2: 6652},
        "decisions": {"GOOD": 9400, "BAD": 7230},
    },
    "E5": {
        "rows": 12472,
        "images": {1: 9978, 2: 2494},
        "decisions": {"GOOD": 7321, "BAD": 5151},
    },
}


class ReplicaDataError(ValueError):
    """Raised when a replica dataset differs from its frozen contract."""


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplicaDataError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict) or set(row) != {"images", "messages"}:
            raise ReplicaDataError(f"invalid row fields at {path}:{line_number}")
        rows.append(row)
    return rows


def decision(row: dict) -> str:
    messages = row["messages"]
    if (
        not isinstance(messages, list)
        or len(messages) != 3
        or [message.get("role") for message in messages]
        != ["system", "user", "assistant"]
    ):
        raise ReplicaDataError("every row must contain system/user/assistant messages")
    try:
        payload = json.loads(messages[-1]["content"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ReplicaDataError("assistant label is not JSON") from exc
    if set(payload) != {"decision", "categories", "reasons"}:
        raise ReplicaDataError(f"unexpected label keys: {sorted(payload)}")
    value = payload["decision"]
    if value not in {"GOOD", "BAD"}:
        raise ReplicaDataError(f"invalid decision: {value}")
    return value


def validate(experiment: str, train_path: Path, dev_path: Path) -> dict:
    contract = CONTRACTS[experiment]
    train = load_rows(train_path)
    dev = load_rows(dev_path)
    train_images = Counter(len(row["images"]) for row in train)
    train_decisions = Counter(decision(row) for row in train)
    dev_decisions = Counter(decision(row) for row in dev)
    if len(train) != contract["rows"]:
        raise ReplicaDataError(f"unexpected Train rows: {len(train)}")
    if train_images != contract["images"]:
        raise ReplicaDataError(f"unexpected Train image counts: {dict(train_images)}")
    if train_decisions != contract["decisions"]:
        raise ReplicaDataError(f"unexpected Train decisions: {dict(train_decisions)}")
    if len(dev) != 200 or dev_decisions != {"GOOD": 142, "BAD": 58}:
        raise ReplicaDataError(
            f"unexpected corrected Dev: rows={len(dev)} decisions={dict(dev_decisions)}"
        )
    dev_sha256 = hashlib.sha256(dev_path.read_bytes()).hexdigest()
    if dev_sha256 != DEV_SHA256:
        raise ReplicaDataError(f"corrected Dev sha256 mismatch: {dev_sha256}")
    train_sources = {row["images"][0] for row in train}
    dev_sources = {row["images"][0] for row in dev}
    if len(train_sources) != 8026 or len(dev_sources) != 200:
        raise ReplicaDataError(
            f"unexpected unique sources: train={len(train_sources)} dev={len(dev_sources)}"
        )
    if train_sources & dev_sources:
        raise ReplicaDataError("Train/Dev image overlap")
    missing = [
        image
        for image in {image for row in train + dev for image in row["images"]}
        if not Path(image).is_file()
    ]
    if missing:
        raise ReplicaDataError(f"missing images: {len(missing)}; first={missing[0]}")
    return {
        "experiment": experiment,
        "train_rows": len(train),
        "train_images": dict(train_images),
        "train_decisions": dict(train_decisions),
        "dev_decisions": dict(dev_decisions),
        "dev_sha256": dev_sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=sorted(CONTRACTS))
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--dev", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = validate(args.experiment, args.train, args.dev)
    except (OSError, ReplicaDataError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    print(f"QWEN36_{args.experiment}_DATA_PREFLIGHT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
