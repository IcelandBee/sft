#!/usr/bin/env python3
"""Build the deterministic E4 T1/T2/T3 crop-aux ms-swift dataset."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import tempfile

from PIL import Image

try:
    from scripts.audit_e4_bbox_coverage import BboxAuditError, _validate_instance, load_jsonl
    from scripts.build_e4_token_poc import LOCAL_PROMPT, SYSTEM_PROMPT, _crop_box
except ModuleNotFoundError:  # Support absolute-path execution on the server.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.audit_e4_bbox_coverage import BboxAuditError, _validate_instance, load_jsonl
    from scripts.build_e4_token_poc import LOCAL_PROMPT, SYSTEM_PROMPT, _crop_box


class E4DatasetError(ValueError):
    """Raised when E4 sources or generated rows violate the locked protocol."""


def adaptive_crop_scale(area_ratio: float) -> tuple[float, str]:
    """Return the locked single-crop scale and its audit bucket."""
    if not 0 < area_ratio <= 1:
        raise E4DatasetError(f"invalid bbox area ratio: {area_ratio}")
    if area_ratio <= 0.01:
        return 2.0, "small_2.0x"
    scale = min(1.5, math.sqrt(0.70 / area_ratio))
    scale = max(1.0, scale)
    if math.isclose(scale, 1.5):
        return 1.5, "standard_1.5x"
    if math.isclose(scale, 1.0):
        return 1.0, "very_large_1.0x"
    return scale, "large_adaptive"


def _payload(row: dict, location: str) -> dict:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise E4DatasetError(f"{location} must have three messages")
    if [item.get("role") for item in messages] != ["system", "user", "assistant"]:
        raise E4DatasetError(f"{location} has invalid message roles")
    try:
        payload = json.loads(messages[-1]["content"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise E4DatasetError(f"{location} has invalid assistant JSON") from exc
    if not isinstance(payload, dict) or payload.get("decision") not in {"GOOD", "BAD"}:
        raise E4DatasetError(f"{location} has invalid decision")
    return payload


def _ms_image(row: dict, location: str) -> str:
    images = row.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
        raise E4DatasetError(f"{location} must contain one image")
    return images[0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    ).encode("utf-8")


def _local_row(full_image: str, crop_image: str, payload: dict) -> dict:
    return {
        "images": [full_image, crop_image],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": LOCAL_PROMPT},
            {
                "role": "assistant",
                "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            },
        ],
    }


def _matched_normal_box(
    width: int,
    height: int,
    width_ratio: float,
    height_ratio: float,
    quadrant: int,
) -> tuple[int, int, int, int]:
    target_width = min(width, max(1, round(width * width_ratio)))
    target_height = min(height, max(1, round(height * height_ratio)))
    center_x = width * (0.25 if quadrant % 2 == 0 else 0.75)
    center_y = height * (0.25 if quadrant < 2 else 0.75)
    left = min(max(0, round(center_x - target_width / 2)), width - target_width)
    top = min(max(0, round(center_y - target_height / 2)), height - target_height)
    return left, top, left + target_width, top + target_height


def _collect_candidates(label_rows: list[dict], max_per_bad_image: int) -> tuple[list[dict], list[dict]]:
    candidates: list[dict] = []
    good_rows: list[dict] = []
    seen_keys: set[str] = set()
    seen_paths: set[str] = set()
    for row_number, row in enumerate(label_rows, start=1):
        image_key = row.get("image_key")
        image_path_value = row.get("image_path")
        decision = row.get("decision")
        instances = row.get("instances")
        if not isinstance(image_key, str) or not image_key or image_key in seen_keys:
            raise E4DatasetError(f"invalid or duplicate image_key at label row {row_number}")
        if not isinstance(image_path_value, str) or not image_path_value or image_path_value in seen_paths:
            raise E4DatasetError(f"invalid or duplicate image_path at label row {row_number}")
        if decision not in {"GOOD", "BAD"} or not isinstance(instances, list):
            raise E4DatasetError(f"invalid label row {row_number}")
        seen_keys.add(image_key)
        seen_paths.add(image_path_value)
        image_path = Path(image_path_value)
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except OSError as exc:
            raise E4DatasetError(f"cannot read label image: {image_path}") from exc
        if decision == "GOOD":
            if instances:
                raise E4DatasetError(f"GOOD label row {row_number} has instances")
            good_rows.append(row)
            continue
        if not instances:
            raise E4DatasetError(f"BAD label row {row_number} has no instances")
        per_image: list[dict] = []
        for instance_index, instance in enumerate(instances):
            try:
                bbox, category = _validate_instance(
                    instance, f"label row {row_number}.instances[{instance_index}]"
                )
            except BboxAuditError as exc:
                raise E4DatasetError(str(exc)) from exc
            if category == "其他":
                raise E4DatasetError(f"broad-clean Train contains 其他 at row {row_number}")
            if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > width or bbox[3] > height:
                raise E4DatasetError(f"out-of-bounds bbox at label row {row_number}")
            area_ratio = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (width * height)
            scale, scale_bucket = adaptive_crop_scale(area_ratio)
            crop_box = _crop_box(bbox, width, height, scale)
            candidate_id = f"{image_key}#{instance_index}"
            per_image.append(
                {
                    "candidate_id": candidate_id,
                    "image_key": image_key,
                    "image_path": image_path_value,
                    "instance_index": instance_index,
                    "bbox": bbox,
                    "category": category,
                    "reason": instance["reason"].strip(),
                    "area_ratio": area_ratio,
                    "scale": scale,
                    "scale_bucket": scale_bucket,
                    "crop_box": crop_box,
                    "crop_width_ratio": (crop_box[2] - crop_box[0]) / width,
                    "crop_height_ratio": (crop_box[3] - crop_box[1]) / height,
                }
            )
        # Prefer the smallest visible structures when a multi-instance image exceeds the cap.
        per_image.sort(key=lambda item: (item["area_ratio"], item["instance_index"]))
        candidates.extend(per_image[:max_per_bad_image])
    return candidates, good_rows


def _balanced_repeat(candidates: list[dict], target: int, seed: int) -> list[dict]:
    if not candidates:
        raise E4DatasetError("no abnormal crop candidates")
    rng = random.Random(seed)
    selected: list[dict] = []
    while len(selected) < target:
        cycle = list(candidates)
        rng.shuffle(cycle)
        selected.extend(cycle[: target - len(selected)])
    return selected


def build_e4_dataset(
    *,
    label_rows: list[dict],
    t1_rows: list[dict],
    dev_bytes: bytes,
    dev_rows: list[dict],
    output_dir: Path,
    label_sha256: str,
    t1_sha256: str,
    dev_sha256: str,
    local_bad_target: int = 3326,
    local_good_target: int = 3326,
    max_per_bad_image: int = 2,
    seed: int = 42,
) -> dict:
    if output_dir.exists():
        raise E4DatasetError(f"output directory already exists: {output_dir}")
    if local_bad_target != local_good_target:
        raise E4DatasetError("local BAD and GOOD targets must be equal")

    label_by_path = {row["image_path"]: row for row in label_rows}
    t1_paths: set[str] = set()
    t1_decisions: Counter[str] = Counter()
    for row_number, row in enumerate(t1_rows, start=1):
        image = _ms_image(row, f"T1 row {row_number}")
        payload = _payload(row, f"T1 row {row_number}")
        if image not in label_by_path:
            raise E4DatasetError(f"T1 image absent from broad-clean labels: {image}")
        if payload["decision"] != label_by_path[image]["decision"]:
            raise E4DatasetError(f"T1 decision differs from labels: {image}")
        t1_paths.add(image)
        t1_decisions[payload["decision"]] += 1
    if t1_paths != set(label_by_path):
        raise E4DatasetError("T1 unique image set differs from broad-clean labels")

    dev_paths = {_ms_image(row, f"Dev row {index}") for index, row in enumerate(dev_rows, 1)}
    if t1_paths & dev_paths:
        raise E4DatasetError("Train and Dev image sets overlap")

    candidates, good_labels = _collect_candidates(label_rows, max_per_bad_image)
    selected_bad = _balanced_repeat(candidates, local_bad_target, seed)
    if len(good_labels) < local_good_target:
        raise E4DatasetError("not enough unique GOOD images for matched T3 crops")
    good_pool = list(good_labels)
    random.Random(seed + 1).shuffle(good_pool)
    selected_good = good_pool[:local_good_target]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    entries: list[dict] = []
    local_manifest_by_entry: dict[int, dict] = {}
    crop_bytes_total = 0
    try:
        (staging / "crops" / "t2_bad").mkdir(parents=True)
        (staging / "crops" / "t3_good").mkdir(parents=True)

        crop_path_by_candidate: dict[str, str] = {}
        for candidate in candidates:
            digest = hashlib.sha256(candidate["candidate_id"].encode("utf-8")).hexdigest()[:20]
            filename = f"{digest}.png"
            staging_crop = staging / "crops" / "t2_bad" / filename
            final_crop = output_dir / "crops" / "t2_bad" / filename
            with Image.open(candidate["image_path"]) as image:
                image.crop(candidate["crop_box"]).convert("RGB").save(staging_crop)
            crop_bytes_total += staging_crop.stat().st_size
            crop_path_by_candidate[candidate["candidate_id"]] = str(final_crop)

        occurrence: Counter[str] = Counter()
        bad_entry_indices: list[int] = []
        for candidate in selected_bad:
            occurrence[candidate["candidate_id"]] += 1
            payload = {
                "decision": "BAD",
                "categories": [candidate["category"]],
                "reasons": [candidate["reason"]],
            }
            entry_index = len(entries)
            entries.append(
                {
                    "sample_type": "T2_BAD",
                    "row": _local_row(
                        candidate["image_path"],
                        crop_path_by_candidate[candidate["candidate_id"]],
                        payload,
                    ),
                }
            )
            bad_entry_indices.append(entry_index)
            local_manifest_by_entry[entry_index] = {
                "sample_type": "T2_BAD",
                "image_key": candidate["image_key"],
                "source_image": candidate["image_path"],
                "crop_image": crop_path_by_candidate[candidate["candidate_id"]],
                "candidate_id": candidate["candidate_id"],
                "candidate_occurrence": occurrence[candidate["candidate_id"]],
                "bbox": list(candidate["bbox"]),
                "bbox_area_ratio": candidate["area_ratio"],
                "crop_box": list(candidate["crop_box"]),
                "crop_width_ratio": candidate["crop_width_ratio"],
                "crop_height_ratio": candidate["crop_height_ratio"],
                "crop_scale": candidate["scale"],
                "scale_bucket": candidate["scale_bucket"],
                "category": candidate["category"],
                "reason": candidate["reason"],
            }

        for pair_index, (good, candidate) in enumerate(zip(selected_good, selected_bad)):
            source_path = Path(good["image_path"])
            digest = hashlib.sha256(
                f"{good['image_key']}|{pair_index}|{candidate['candidate_id']}".encode("utf-8")
            ).digest()
            quadrant = digest[0] % 4
            filename = hashlib.sha256(
                f"{good['image_key']}|{pair_index}".encode("utf-8")
            ).hexdigest()[:20] + ".png"
            staging_crop = staging / "crops" / "t3_good" / filename
            final_crop = output_dir / "crops" / "t3_good" / filename
            with Image.open(source_path) as image:
                width, height = image.size
                crop_box = _matched_normal_box(
                    width,
                    height,
                    candidate["crop_width_ratio"],
                    candidate["crop_height_ratio"],
                    quadrant,
                )
                image.crop(crop_box).convert("RGB").save(staging_crop)
            crop_bytes_total += staging_crop.stat().st_size
            actual_width_ratio = (crop_box[2] - crop_box[0]) / width
            actual_height_ratio = (crop_box[3] - crop_box[1]) / height
            entry_index = len(entries)
            entries.append(
                {
                    "sample_type": "T3_GOOD",
                    "row": _local_row(
                        str(source_path),
                        str(final_crop),
                        {"decision": "GOOD", "categories": [], "reasons": []},
                    ),
                }
            )
            local_manifest_by_entry[entry_index] = {
                "sample_type": "T3_GOOD",
                "image_key": good["image_key"],
                "source_image": str(source_path),
                "crop_image": str(final_crop),
                "crop_box": list(crop_box),
                "quadrant": quadrant,
                "crop_width_ratio": actual_width_ratio,
                "crop_height_ratio": actual_height_ratio,
                "matched_t2_candidate_id": candidate["candidate_id"],
                "matched_width_ratio_error": abs(
                    actual_width_ratio - candidate["crop_width_ratio"]
                ),
                "matched_height_ratio_error": abs(
                    actual_height_ratio - candidate["crop_height_ratio"]
                ),
            }

        t1_entries = [{"sample_type": "T1_FULL", "row": row} for row in t1_rows]
        entries = t1_entries + entries
        # Local manifest keys were based on the pre-T1 local list; shift before shuffling.
        local_manifest_by_entry = {
            index + len(t1_entries): value for index, value in local_manifest_by_entry.items()
        }
        order = list(range(len(entries)))
        random.Random(seed).shuffle(order)
        output_rows: list[dict] = []
        local_manifest_rows: list[dict] = []
        type_counts: Counter[str] = Counter()
        for output_index, entry_index in enumerate(order):
            entry = entries[entry_index]
            output_rows.append(entry["row"])
            type_counts[entry["sample_type"]] += 1
            if entry_index in local_manifest_by_entry:
                local_manifest_rows.append(
                    {
                        "train_output_index": output_index,
                        **local_manifest_by_entry[entry_index],
                    }
                )

        if type_counts != Counter(
            {"T1_FULL": len(t1_rows), "T2_BAD": local_bad_target, "T3_GOOD": local_good_target}
        ):
            raise E4DatasetError(f"unexpected mixed counts: {dict(type_counts)}")
        if len(local_manifest_rows) != local_bad_target + local_good_target:
            raise E4DatasetError("local manifest count mismatch")

        train_bytes = _jsonl_bytes(output_rows)
        local_manifest_bytes = _jsonl_bytes(local_manifest_rows)
        (staging / "train.jsonl").write_bytes(train_bytes)
        (staging / "dev.jsonl").write_bytes(dev_bytes)
        (staging / "local_manifest.jsonl").write_bytes(local_manifest_bytes)

        unique_selected_bad = {item["candidate_id"] for item in selected_bad}
        category_unique: Counter[str] = Counter(item["category"] for item in candidates)
        category_rows: Counter[str] = Counter(item["category"] for item in selected_bad)
        scale_unique: Counter[str] = Counter(item["scale_bucket"] for item in candidates)
        scale_rows: Counter[str] = Counter(item["scale_bucket"] for item in selected_bad)
        max_width_error = max(
            (
                row["matched_width_ratio_error"]
                for row in local_manifest_rows
                if row["sample_type"] == "T3_GOOD"
            ),
            default=0.0,
        )
        max_height_error = max(
            (
                row["matched_height_ratio_error"]
                for row in local_manifest_rows
                if row["sample_type"] == "T3_GOOD"
            ),
            default=0.0,
        )
        summary = {
            "protocol_version": "e4_crop_aux_dataset_v1",
            "seed": seed,
            "source_scope": "broad_clean_train_plus_adjudicated_dev",
            "test_untouched": True,
            "inputs": {
                "label_rows": len(label_rows),
                "label_sha256": label_sha256,
                "t1_rows": len(t1_rows),
                "t1_sha256": t1_sha256,
                "t1_decisions": dict(sorted(t1_decisions.items())),
                "dev_rows": len(dev_rows),
                "dev_sha256": dev_sha256,
            },
            "output": {
                "train_rows": len(output_rows),
                "sample_type_counts": dict(sorted(type_counts.items())),
                "train_sha256": hashlib.sha256(train_bytes).hexdigest(),
                "dev_sha256": hashlib.sha256(dev_bytes).hexdigest(),
                "local_manifest_sha256": hashlib.sha256(local_manifest_bytes).hexdigest(),
                "unique_t2_crop_files": len(candidates),
                "t3_crop_files": local_good_target,
                "crop_files_total": len(candidates) + local_good_target,
                "crop_bytes_total": crop_bytes_total,
            },
            "t2_sampling": {
                "valid_candidates_before_image_cap": sum(len(row["instances"]) for row in label_rows if row["decision"] == "BAD"),
                "candidates_after_cap": len(candidates),
                "selected_rows": len(selected_bad),
                "unique_selected_candidates": len(unique_selected_bad),
                "repeated_rows": len(selected_bad) - len(unique_selected_bad),
                "oversample_factor": len(selected_bad) / len(unique_selected_bad),
                "max_crops_per_bad_image": max_per_bad_image,
                "category_unique_candidates": dict(sorted(category_unique.items())),
                "category_selected_rows": dict(sorted(category_rows.items())),
                "scale_unique_candidates": dict(sorted(scale_unique.items())),
                "scale_selected_rows": dict(sorted(scale_rows.items())),
            },
            "t3_matching": {
                "unique_good_sources": len(selected_good),
                "max_width_ratio_error": max_width_error,
                "max_height_ratio_error": max_height_error,
            },
            "training_contract": {
                "max_length": 3072,
                "image_max_token_num": 1024,
                "max_steps": 1248,
                "local_bad_to_good": "1:1",
                "t1_to_local": "60:40",
            },
            "status": "PASS",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--t1-train", required=True, type=Path)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-label-rows", type=int, default=8026)
    parser.add_argument("--expected-t1-rows", type=int, default=9978)
    parser.add_argument("--expected-dev-rows", type=int, default=200)
    parser.add_argument("--expected-label-good", type=int)
    parser.add_argument("--expected-label-bad", type=int)
    parser.add_argument("--expected-t1-good", type=int)
    parser.add_argument("--expected-t1-bad", type=int)
    parser.add_argument("--expected-dev-good", type=int)
    parser.add_argument("--expected-dev-bad", type=int)
    parser.add_argument("--expected-dev-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        labels = load_jsonl(args.labels)
        t1 = load_jsonl(args.t1_train)
        dev = load_jsonl(args.dev)
        for name, rows, expected in (
            ("labels", labels, args.expected_label_rows),
            ("T1", t1, args.expected_t1_rows),
            ("Dev", dev, args.expected_dev_rows),
        ):
            if len(rows) != expected:
                raise E4DatasetError(f"expected {expected} {name} rows, got {len(rows)}")
        decision_checks = (
            (
                "labels",
                Counter(row.get("decision") for row in labels),
                args.expected_label_good,
                args.expected_label_bad,
            ),
            (
                "T1",
                Counter(_payload(row, f"T1 row {index}")["decision"] for index, row in enumerate(t1, 1)),
                args.expected_t1_good,
                args.expected_t1_bad,
            ),
            (
                "Dev",
                Counter(_payload(row, f"Dev row {index}")["decision"] for index, row in enumerate(dev, 1)),
                args.expected_dev_good,
                args.expected_dev_bad,
            ),
        )
        for name, actual, expected_good, expected_bad in decision_checks:
            if expected_good is not None and actual["GOOD"] != expected_good:
                raise E4DatasetError(
                    f"expected {expected_good} {name} GOOD rows, got {actual['GOOD']}"
                )
            if expected_bad is not None and actual["BAD"] != expected_bad:
                raise E4DatasetError(
                    f"expected {expected_bad} {name} BAD rows, got {actual['BAD']}"
                )
            if set(actual) - {"GOOD", "BAD"}:
                raise E4DatasetError(f"{name} contains invalid decisions: {dict(actual)}")
        actual_dev_sha256 = _sha256(args.dev)
        if (
            args.expected_dev_sha256 is not None
            and actual_dev_sha256 != args.expected_dev_sha256
        ):
            raise E4DatasetError(
                "corrected Dev sha256 mismatch: "
                f"expected {args.expected_dev_sha256}, got {actual_dev_sha256}"
            )
        dev_bytes = args.dev.read_bytes()
        summary = build_e4_dataset(
            label_rows=labels,
            t1_rows=t1,
            dev_bytes=dev_bytes,
            dev_rows=dev,
            output_dir=args.output_dir,
            label_sha256=_sha256(args.labels),
            t1_sha256=_sha256(args.t1_train),
            dev_sha256=actual_dev_sha256,
        )
    except (BboxAuditError, E4DatasetError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("=== E4 CROP-AUX DATASET ===")
    print(f"sample_types={summary['output']['sample_type_counts']}")
    print(f"t2_sampling={summary['t2_sampling']}")
    print(f"t3_matching={summary['t3_matching']}")
    print(f"crop_files={summary['output']['crop_files_total']}")
    print(f"train_sha256={summary['output']['train_sha256']}")
    print(f"output={args.output_dir}")
    print("E4_CROP_AUX_BUILD: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
