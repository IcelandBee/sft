#!/usr/bin/env python3
"""Read-only bbox coverage audit for the E4 crop-aux Train design."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable

from PIL import Image


class BboxAuditError(ValueError):
    """Raised when the broad-clean Train contract is invalid."""


def load_jsonl(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise BboxAuditError(f"cannot read Train JSONL: {path}") from exc
    rows: list[dict] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BboxAuditError(f"invalid JSON at row {line_number}") from exc
        if not isinstance(row, dict):
            raise BboxAuditError(f"row {line_number} must be an object")
        rows.append(row)
    return rows


def _ranked(counter: Counter[str]) -> list[dict]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _quantile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _quantiles(values: list[float]) -> dict[str, float | None]:
    return {
        name: _quantile(values, probability)
        for name, probability in (
            ("min", 0.0),
            ("p10", 0.10),
            ("p25", 0.25),
            ("p50", 0.50),
            ("p75", 0.75),
            ("p90", 0.90),
            ("p95", 0.95),
            ("max", 1.0),
        )
    }


def _area_bucket(area_ratio: float) -> str:
    if area_ratio <= 0.0025:
        return "<=0.25%"
    if area_ratio <= 0.01:
        return "0.25%-1%"
    if area_ratio <= 0.04:
        return "1%-4%"
    if area_ratio <= 0.16:
        return "4%-16%"
    return ">16%"


def _expanded_area_ratio(
    bbox: tuple[float, float, float, float], width: int, height: int, scale: float
) -> float:
    x1, y1, x2, y2 = bbox
    crop_width = min(float(width), (x2 - x1) * scale)
    crop_height = min(float(height), (y2 - y1) * scale)
    return crop_width * crop_height / (width * height)


def _validate_instance(instance: object, location: str) -> tuple[tuple[float, ...], str]:
    if not isinstance(instance, dict):
        raise BboxAuditError(f"{location} must be an object")
    category = instance.get("category")
    reason = instance.get("reason")
    bbox = instance.get("bbox")
    if not isinstance(category, str) or not category.strip():
        raise BboxAuditError(f"{location}.category must be non-empty")
    if not isinstance(reason, str) or not reason.strip():
        raise BboxAuditError(f"{location}.reason must be non-empty")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise BboxAuditError(f"{location}.bbox must contain four coordinates")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox):
        raise BboxAuditError(f"{location}.bbox coordinates must be numeric")
    coordinates = tuple(float(value) for value in bbox)
    if not all(math.isfinite(value) for value in coordinates):
        raise BboxAuditError(f"{location}.bbox coordinates must be finite")
    if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
        raise BboxAuditError(f"{location}.bbox must satisfy x2>x1 and y2>y1")
    return coordinates, category.strip()


def audit_bbox_coverage(
    rows: list[dict],
    *,
    source_sha256: str,
    small_area_threshold: float = 0.01,
    max_crops_per_bad_image: int = 2,
    t1_rows: int = 9978,
    local_share: float = 0.40,
) -> dict:
    """Validate Train and return deterministic bbox/crop feasibility statistics."""
    if not 0 < small_area_threshold < 1:
        raise BboxAuditError("small_area_threshold must be between 0 and 1")
    if max_crops_per_bad_image < 1:
        raise BboxAuditError("max_crops_per_bad_image must be positive")
    if t1_rows < 1 or not 0 < local_share < 1:
        raise BboxAuditError("invalid sampling projection parameters")

    decision_counts: Counter[str] = Counter()
    category_instances: Counter[str] = Counter()
    category_images: Counter[str] = Counter()
    instances_per_bad_image: Counter[str] = Counter()
    bbox_area_ratios: list[float] = []
    crop_15_area_ratios: list[float] = []
    crop_20_area_ratios: list[float] = []
    bbox_area_buckets: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    issue_examples: list[dict] = []
    seen_keys: set[str] = set()
    seen_paths: set[str] = set()
    bad_candidate_counts: list[int] = []
    small_instances = 0

    def issue(code: str, row_number: int, detail: str) -> None:
        issue_counts[code] += 1
        if len(issue_examples) < 50:
            issue_examples.append({"row": row_number, "code": code, "detail": detail})

    for row_number, row in enumerate(rows, start=1):
        decision = row.get("decision")
        image_key = row.get("image_key")
        image_path_value = row.get("image_path")
        instances = row.get("instances")
        if decision not in {"GOOD", "BAD"}:
            raise BboxAuditError(f"row {row_number} has invalid decision")
        if not isinstance(image_key, str) or not image_key.strip():
            raise BboxAuditError(f"row {row_number} has invalid image_key")
        if image_key in seen_keys:
            raise BboxAuditError(f"duplicate image_key at row {row_number}: {image_key}")
        seen_keys.add(image_key)
        if not isinstance(image_path_value, str) or not image_path_value.strip():
            raise BboxAuditError(f"row {row_number} has invalid image_path")
        if image_path_value in seen_paths:
            raise BboxAuditError(f"duplicate image_path at row {row_number}: {image_path_value}")
        seen_paths.add(image_path_value)
        if not isinstance(instances, list):
            raise BboxAuditError(f"row {row_number}.instances must be a list")
        if decision == "GOOD" and instances:
            raise BboxAuditError(f"GOOD row {row_number} must not contain instances")
        if decision == "BAD" and not instances:
            raise BboxAuditError(f"BAD row {row_number} must contain instances")
        decision_counts[decision] += 1

        image_path = Path(image_path_value)
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except (OSError, ValueError) as exc:
            issue("image_unreadable", row_number, f"{image_path}: {exc}")
            continue
        if width <= 0 or height <= 0:
            issue("image_invalid_dimensions", row_number, f"{width}x{height}")
            continue

        if decision == "GOOD":
            continue

        image_categories: set[str] = set()
        candidate_count = 0
        for instance_index, instance in enumerate(instances):
            location = f"row {row_number}.instances[{instance_index}]"
            bbox, category = _validate_instance(instance, location)
            if category == "其他":
                issue("forbidden_other_category", row_number, location)
                continue
            x1, y1, x2, y2 = bbox
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                issue(
                    "bbox_out_of_bounds",
                    row_number,
                    f"{list(bbox)} outside {width}x{height}",
                )
                continue
            area_ratio = (x2 - x1) * (y2 - y1) / (width * height)
            bbox_area_ratios.append(area_ratio)
            crop_15_area_ratios.append(_expanded_area_ratio(bbox, width, height, 1.5))
            crop_20_area_ratios.append(_expanded_area_ratio(bbox, width, height, 2.0))
            bbox_area_buckets[_area_bucket(area_ratio)] += 1
            category_instances[category] += 1
            image_categories.add(category)
            candidate_count += 1
            if area_ratio <= small_area_threshold:
                small_instances += 1
                candidate_count += 1
        category_images.update(image_categories)
        instance_total = len(instances)
        bucket = "4+" if instance_total >= 4 else str(instance_total)
        instances_per_bad_image[bucket] += 1
        bad_candidate_counts.append(candidate_count)

    local_total_target = round(t1_rows * local_share / (1 - local_share))
    local_bad_target = local_total_target // 2
    local_good_target = local_total_target - local_bad_target
    uncapped_bad_candidates = sum(bad_candidate_counts)
    capped_bad_candidates = sum(
        min(count, max_crops_per_bad_image) for count in bad_candidate_counts
    )
    good_images = decision_counts["GOOD"]

    return {
        "protocol_version": "e4_bbox_coverage_audit_v1",
        "source_sha256": source_sha256,
        "source_scope": "broad_clean_train_only",
        "test_untouched": True,
        "dev_untouched": True,
        "rows": len(rows),
        "unique_image_keys": len(seen_keys),
        "unique_image_paths": len(seen_paths),
        "decision_counts": dict(sorted(decision_counts.items())),
        "valid_bbox_instances": len(bbox_area_ratios),
        "small_area_threshold": small_area_threshold,
        "small_bbox_instances": small_instances,
        "bbox_area_ratio_quantiles": _quantiles(bbox_area_ratios),
        "bbox_area_ratio_buckets": [
            {"value": name, "count": bbox_area_buckets[name]}
            for name in ("<=0.25%", "0.25%-1%", "1%-4%", "4%-16%", ">16%")
        ],
        "crop_1_5_area_ratio_quantiles": _quantiles(crop_15_area_ratios),
        "crop_2_0_area_ratio_quantiles": _quantiles(crop_20_area_ratios),
        "bad_images_by_instance_count": [
            {"value": name, "count": instances_per_bad_image[name]}
            for name in ("1", "2", "3", "4+")
        ],
        "category_instance_counts": _ranked(category_instances),
        "category_image_counts": _ranked(category_images),
        "crop_projection": {
            "t1_rows": t1_rows,
            "local_share": local_share,
            "local_total_target": local_total_target,
            "local_bad_target": local_bad_target,
            "local_good_target": local_good_target,
            "abnormal_candidates_uncapped": uncapped_bad_candidates,
            "abnormal_candidates_cap_per_image": max_crops_per_bad_image,
            "abnormal_candidates_capped": capped_bad_candidates,
            "abnormal_oversample_factor_if_capped": (
                local_bad_target / capped_bad_candidates if capped_bad_candidates else None
            ),
            "normal_candidates_one_per_good_image": good_images,
            "normal_candidates_two_per_good_image": good_images * 2,
        },
        "issues": {
            "total": sum(issue_counts.values()),
            "counts": dict(sorted(issue_counts.items())),
            "examples_first_50": issue_examples,
        },
        "status": "PASS" if not issue_counts else "NEEDS_REVIEW",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-good", type=int)
    parser.add_argument("--expected-bad", type=int)
    parser.add_argument("--small-area-threshold", type=float, default=0.01)
    parser.add_argument("--max-crops-per-bad-image", type=int, default=2)
    parser.add_argument("--t1-rows", type=int, default=9978)
    parser.add_argument("--local-share", type=float, default=0.40)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        print(f"ERROR: output already exists: {args.output}", file=sys.stderr)
        return 2
    try:
        source_bytes = args.train.read_bytes()
        rows = load_jsonl(args.train)
        summary = audit_bbox_coverage(
            rows,
            source_sha256=hashlib.sha256(source_bytes).hexdigest(),
            small_area_threshold=args.small_area_threshold,
            max_crops_per_bad_image=args.max_crops_per_bad_image,
            t1_rows=args.t1_rows,
            local_share=args.local_share,
        )
        expected = {
            "rows": args.expected_rows,
            "GOOD": args.expected_good,
            "BAD": args.expected_bad,
        }
        actual = {
            "rows": summary["rows"],
            "GOOD": summary["decision_counts"].get("GOOD", 0),
            "BAD": summary["decision_counts"].get("BAD", 0),
        }
        for name, expected_value in expected.items():
            if expected_value is not None and actual[name] != expected_value:
                raise BboxAuditError(
                    f"expected {name}={expected_value}, got {actual[name]}"
                )
    except (BboxAuditError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    projection = summary["crop_projection"]
    print("=== E4 BBOX COVERAGE ===")
    print(f"rows={summary['rows']} decisions={summary['decision_counts']}")
    print(
        f"valid_instances={summary['valid_bbox_instances']} "
        f"small_instances={summary['small_bbox_instances']}"
    )
    print(f"area_buckets={summary['bbox_area_ratio_buckets']}")
    print(f"category_instances={summary['category_instance_counts']}")
    print(
        f"local_target={projection['local_total_target']} "
        f"BAD/GOOD={projection['local_bad_target']}/{projection['local_good_target']}"
    )
    print(
        f"abnormal_candidates={projection['abnormal_candidates_uncapped']} "
        f"capped={projection['abnormal_candidates_capped']} "
        f"oversample={projection['abnormal_oversample_factor_if_capped']}"
    )
    print(f"issues={summary['issues']['total']} status={summary['status']}")
    print(f"summary={args.output}")
    if summary["status"] != "PASS":
        print("E4_BBOX_AUDIT: NEEDS_REVIEW")
        return 2
    print("E4_BBOX_AUDIT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
