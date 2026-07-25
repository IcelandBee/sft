#!/usr/bin/env python3
"""Map the 250 final Test labels to uploaded images and audit Train/Dev leakage."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from xml.etree import ElementTree
from zipfile import ZipFile

try:
    from scripts.inspect_final_test250_sources import (
        IMAGE_SUFFIXES,
        MAIN_NS,
        file_sha256,
        parse_cell_reference,
        parse_cell_value,
        parse_shared_strings,
        workbook_sheet_paths,
    )
except ModuleNotFoundError:  # Support absolute-path execution on the server.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.inspect_final_test250_sources import (
        IMAGE_SUFFIXES,
        MAIN_NS,
        file_sha256,
        parse_cell_reference,
        parse_cell_value,
        parse_shared_strings,
        workbook_sheet_paths,
    )


class TestIndependenceError(ValueError):
    """Raised when Test250 mapping or source isolation is ambiguous."""


def load_sheet_rows(workbook_path: Path) -> list[dict[int, object]]:
    try:
        archive = ZipFile(workbook_path)
    except OSError as exc:
        raise TestIndependenceError(f"cannot open workbook: {exc}") from exc
    try:
        sheets = workbook_sheet_paths(archive)
        if len(sheets) != 1:
            raise TestIndependenceError(f"expected one worksheet, got {len(sheets)}")
        shared_strings = parse_shared_strings(archive)
        root = ElementTree.fromstring(archive.read(sheets[0]["path"]))
        rows: dict[int, dict[int, object]] = {}
        for cell in root.findall(f".//{{{MAIN_NS}}}c"):
            reference = cell.get("r")
            if not reference:
                continue
            row_number, column = parse_cell_reference(reference)
            value = parse_cell_value(cell, shared_strings)
            if value is not None:
                rows.setdefault(row_number, {})[column] = value
        return [rows[number] for number in sorted(rows)]
    finally:
        archive.close()


def classify_label(value: object) -> str:
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = text
    if isinstance(parsed, list) and len(parsed) == 1:
        text = str(parsed[0]).strip()
    else:
        text = str(parsed).strip()
    if "无异常" in text:
        return "GOOD"
    if "有异常" in text:
        return "BAD"
    raise TestIndependenceError(f"unrecognized binary label: {value!r}")


def load_ms_swift_images(path: Path) -> list[str]:
    images: list[str] = []
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                row_images = row["images"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise TestIndependenceError(f"invalid JSONL row {path}:{number}") from exc
            if (
                not isinstance(row_images, list)
                or len(row_images) != 1
                or not isinstance(row_images[0], str)
            ):
                raise TestIndependenceError(
                    f"expected one image at {path}:{number}"
                )
            images.append(row_images[0])
    return images


def audit_independence(
    *,
    workbook_path: Path,
    image_root: Path,
    train_path: Path,
    dev_path: Path,
) -> dict:
    rows = load_sheet_rows(workbook_path)
    if len(rows) != 251:
        raise TestIndependenceError(f"expected header + 250 rows, got {len(rows)}")
    header = rows[0]
    if header.get(6) != "image_version" or header.get(7) != "group_index":
        raise TestIndependenceError(
            f"unexpected workbook headers: F={header.get(6)!r} G={header.get(7)!r}"
        )

    image_paths = sorted(
        (
            path
            for path in image_root.rglob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )
    by_name = {path.name.casefold(): path for path in image_paths}
    train_images = load_ms_swift_images(train_path)
    dev_images = load_ms_swift_images(dev_path)
    train_basenames = {Path(path).name.casefold() for path in train_images}
    dev_basenames = {Path(path).name.casefold() for path in dev_images}
    train_resolved = {str(Path(path).resolve()) for path in train_images}
    dev_resolved = {str(Path(path).resolve()) for path in dev_images}

    mappings: list[dict] = []
    labels: Counter[str] = Counter()
    selected_paths: set[str] = set()
    selected_manifest_digest = hashlib.sha256()
    for excel_row, row in enumerate(rows[1:], start=2):
        group = str(row.get(7, "")).strip()
        version = str(row.get(6, "")).strip()
        if not group or not version:
            raise TestIndependenceError(
                f"missing group/version at Excel row {excel_row}"
            )
        label = classify_label(row.get(9))
        prefix = group.casefold() + "_"
        target_prefix = f"{group}_{version}_".casefold()
        group_files = sorted(
            name for name in by_name if name.startswith(prefix)
        )
        target_files = [
            name for name in group_files if name.startswith(target_prefix)
        ]
        if len(group_files) != 2 or len(target_files) != 1:
            raise TestIndependenceError(
                f"ambiguous files at Excel row {excel_row}: "
                f"group={group} all={group_files} targets={target_files}"
            )
        target_name = target_files[0]
        peer_name = next(name for name in group_files if name != target_name)
        target_path = by_name[target_name].resolve()
        if str(target_path) in selected_paths:
            raise TestIndependenceError(f"duplicate selected target: {target_path}")
        selected_paths.add(str(target_path))
        labels[label] += 1
        target_digest = file_sha256(target_path)
        selected_manifest_digest.update(
            (
                f"{excel_row}\0{group}\0{label}\0{target_path.name}\0"
                f"{target_path.stat().st_size}\0{target_digest}\n"
            ).encode("utf-8")
        )
        mappings.append(
            {
                "excel_row": excel_row,
                "group_index": group,
                "label": label,
                "target_image": str(target_path),
                "target_name": target_path.name,
                "peer_name": by_name[peer_name].name,
                "target_exact_train_overlap": str(target_path) in train_resolved,
                "target_exact_dev_overlap": str(target_path) in dev_resolved,
                "peer_basename_train_overlap": peer_name in train_basenames,
                "peer_basename_dev_overlap": peer_name in dev_basenames,
            }
        )

    if labels != {"GOOD": 186, "BAD": 64}:
        raise TestIndependenceError(f"unexpected label counts: {dict(labels)}")
    overlap_counts = Counter()
    for row in mappings:
        for field in (
            "target_exact_train_overlap",
            "target_exact_dev_overlap",
            "peer_basename_train_overlap",
            "peer_basename_dev_overlap",
        ):
            overlap_counts[field] += int(row[field])
    leakage_rows = [
        row
        for row in mappings
        if any(
            row[field]
            for field in (
                "target_exact_train_overlap",
                "target_exact_dev_overlap",
                "peer_basename_train_overlap",
                "peer_basename_dev_overlap",
            )
        )
    ]
    independent = not leakage_rows
    return {
        "protocol_version": "final_test250_independence_audit_v1",
        "workbook_sha256": file_sha256(workbook_path),
        "source_train": str(train_path.resolve()),
        "source_dev": str(dev_path.resolve()),
        "train_rows": len(train_images),
        "dev_rows": len(dev_images),
        "test_rows": len(mappings),
        "label_counts": dict(sorted(labels.items())),
        "selected_image_manifest_sha256": selected_manifest_digest.hexdigest(),
        "overlap_counts": dict(sorted(overlap_counts.items())),
        "leakage_group_count": len(leakage_rows),
        "leakage_examples": leakage_rows[:30],
        "mapping_examples": mappings[:10],
        "independent_from_train_dev": independent,
        "model_inference_run": False,
        "status": "PASS" if independent else "LEAKAGE_DETECTED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--dev", required=True, type=Path)
    args = parser.parse_args()
    try:
        summary = audit_independence(
            workbook_path=args.workbook,
            image_root=args.image_root,
            train_path=args.train,
            dev_path=args.dev,
        )
    except (TestIndependenceError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"FINAL_TEST250_INDEPENDENCE_AUDIT: {summary['status']}")
    return 0 if summary["independent_from_train_dev"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
