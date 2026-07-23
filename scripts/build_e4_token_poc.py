#!/usr/bin/env python3
"""Build a small representative two-image E4 dataset for token preflight."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

from PIL import Image

try:
    from scripts.audit_e4_bbox_coverage import BboxAuditError, _validate_instance, load_jsonl
except ModuleNotFoundError:  # Support direct execution via an absolute script path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.audit_e4_bbox_coverage import BboxAuditError, _validate_instance, load_jsonl


SYSTEM_PROMPT = (
    "你是AIGC写实人像质量检测器。请依据图片中可见内容判断是否存在明显的生成异常。"
    "严格只输出指定JSON，不要添加分析、解释或Markdown。"
)
LOCAL_PROMPT = """<image>
<image>
第一张图片是完整图，仅用于理解上下文。
第二张图片是待检查的局部区域。
只判断第二张局部区域是否存在明显生成异常。
输出decision、categories和reasons。decision只能是GOOD或BAD。"""


class TokenPocError(ValueError):
    """Raised when representative E4 samples cannot be built safely."""


def _crop_box(
    bbox: tuple[float, float, float, float], width: int, height: int, scale: float
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    target_width = min(float(width), (x2 - x1) * scale)
    target_height = min(float(height), (y2 - y1) * scale)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    left = min(max(0.0, center_x - target_width / 2), width - target_width)
    top = min(max(0.0, center_y - target_height / 2), height - target_height)
    right = left + target_width
    bottom = top + target_height
    return (
        max(0, math.floor(left)),
        max(0, math.floor(top)),
        min(width, math.ceil(right)),
        min(height, math.ceil(bottom)),
    )


def _normal_grid_box(width: int, height: int, quadrant: int) -> tuple[int, int, int, int]:
    crop_width = math.ceil(width * 0.55)
    crop_height = math.ceil(height * 0.55)
    left = 0 if quadrant % 2 == 0 else width - crop_width
    top = 0 if quadrant < 2 else height - crop_height
    return left, top, left + crop_width, top + crop_height


def _evenly_spaced(items: list[dict], count: int) -> list[dict]:
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]


def select_representative_samples(
    rows: list[dict], *, extra_bad_quantiles: int = 3, good_count: int = 5
) -> tuple[list[dict], list[dict]]:
    """Select category extremes plus global bbox quantiles and spaced GOOD images."""
    candidates: list[dict] = []
    good_rows: list[dict] = []
    for row_number, row in enumerate(rows, start=1):
        decision = row.get("decision")
        image_path = Path(str(row.get("image_path", "")))
        if decision == "GOOD":
            good_rows.append(row)
            continue
        if decision != "BAD":
            raise TokenPocError(f"invalid decision at row {row_number}")
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except OSError as exc:
            raise TokenPocError(f"cannot read image at row {row_number}: {image_path}") from exc
        instances = row.get("instances")
        if not isinstance(instances, list) or not instances:
            raise TokenPocError(f"BAD row {row_number} has no instances")
        for instance_index, instance in enumerate(instances):
            try:
                bbox, category = _validate_instance(
                    instance, f"row {row_number}.instances[{instance_index}]"
                )
            except BboxAuditError as exc:
                raise TokenPocError(str(exc)) from exc
            if category == "其他":
                raise TokenPocError(f"forbidden 其他 instance at row {row_number}")
            if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > width or bbox[3] > height:
                raise TokenPocError(f"out-of-bounds bbox at row {row_number}")
            area_ratio = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (width * height)
            candidates.append(
                {
                    "row_number": row_number,
                    "image_key": row["image_key"],
                    "image_path": str(image_path),
                    "bbox": bbox,
                    "category": category,
                    "reason": instance["reason"].strip(),
                    "area_ratio": area_ratio,
                    "width": width,
                    "height": height,
                }
            )
    if not candidates or not good_rows:
        raise TokenPocError("Train must contain BAD bbox candidates and GOOD images")

    selected: list[dict] = []
    markers: set[tuple[int, tuple[float, ...]]] = set()

    def add(candidate: dict, selection_reason: str) -> None:
        marker = (candidate["row_number"], candidate["bbox"])
        if marker in markers:
            return
        markers.add(marker)
        selected.append({**candidate, "selection_reason": selection_reason})

    by_category: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        by_category[candidate["category"]].append(candidate)
    for category in sorted(by_category):
        ordered = sorted(
            by_category[category],
            key=lambda item: (item["area_ratio"], item["image_key"], item["bbox"]),
        )
        add(ordered[0], f"{category}:min_area")
        add(ordered[-1], f"{category}:max_area")

    global_ordered = sorted(
        candidates,
        key=lambda item: (item["area_ratio"], item["image_key"], item["bbox"]),
    )
    for index in range(1, extra_bad_quantiles + 1):
        probability = index / (extra_bad_quantiles + 1)
        position = round((len(global_ordered) - 1) * probability)
        add(global_ordered[position], f"global_q{probability:.2f}")

    ordered_good = sorted(good_rows, key=lambda item: item["image_key"])
    return selected, _evenly_spaced(ordered_good, good_count)


def build_token_poc(
    rows: list[dict], output_dir: Path, *, source_sha256: str
) -> dict:
    if output_dir.exists():
        raise TokenPocError(f"output directory already exists: {output_dir}")
    bad_samples, good_samples = select_representative_samples(rows)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        (staging / "crops").mkdir()
        dataset_rows: list[dict] = []
        manifest_rows: list[dict] = []
        combined = [("T2_BAD", item) for item in bad_samples] + [
            ("T3_GOOD", item) for item in good_samples
        ]
        for index, (sample_type, item) in enumerate(combined):
            source_path = Path(item["image_path"])
            with Image.open(source_path) as image:
                width, height = image.size
                if sample_type == "T2_BAD":
                    scale = 2.0 if item["area_ratio"] <= 0.01 else 1.5
                    crop_box = _crop_box(item["bbox"], width, height, scale)
                    payload = {
                        "decision": "BAD",
                        "categories": [item["category"]],
                        "reasons": [item["reason"]],
                    }
                    selection_reason = item["selection_reason"]
                    area_ratio = item["area_ratio"]
                else:
                    digest = hashlib.sha256(item["image_key"].encode("utf-8")).digest()
                    quadrant = digest[0] % 4
                    crop_box = _normal_grid_box(width, height, quadrant)
                    scale = None
                    payload = {"decision": "GOOD", "categories": [], "reasons": []}
                    selection_reason = f"spaced_good:grid_{quadrant}"
                    area_ratio = None
                crop_name = f"{index:03d}_{sample_type.lower()}.png"
                final_crop = output_dir / "crops" / crop_name
                image.crop(crop_box).convert("RGB").save(staging / "crops" / crop_name)
            dataset_rows.append(
                {
                    "images": [str(source_path), str(final_crop)],
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": LOCAL_PROMPT},
                        {
                            "role": "assistant",
                            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                        },
                    ],
                }
            )
            manifest_rows.append(
                {
                    "index": index,
                    "sample_type": sample_type,
                    "image_key": item["image_key"],
                    "source_image": str(source_path),
                    "crop_image": str(final_crop),
                    "crop_box": list(crop_box),
                    "bbox": list(item["bbox"]) if sample_type == "T2_BAD" else None,
                    "bbox_area_ratio": area_ratio,
                    "crop_scale": scale,
                    "selection_reason": selection_reason,
                    "payload": payload,
                }
            )
        jsonl_text = "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in dataset_rows
        )
        (staging / "poc.jsonl").write_text(jsonl_text, encoding="utf-8")
        manifest = {
            "protocol_version": "e4_two_image_token_poc_v1",
            "source_scope": "broad_clean_train_only",
            "source_sha256": source_sha256,
            "test_untouched": True,
            "dev_untouched": True,
            "rows": len(dataset_rows),
            "sample_counts": {"T2_BAD": len(bad_samples), "T3_GOOD": len(good_samples)},
            "samples": manifest_rows,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int, default=8026)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        source_bytes = args.train.read_bytes()
        rows = load_jsonl(args.train)
        if len(rows) != args.expected_rows:
            raise TokenPocError(f"expected {args.expected_rows} rows, got {len(rows)}")
        manifest = build_token_poc(
            rows,
            args.output_dir,
            source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        )
    except (BboxAuditError, TokenPocError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"rows={manifest['rows']} samples={manifest['sample_counts']}")
    print(f"poc={args.output_dir / 'poc.jsonl'}")
    print("E4_TOKEN_POC_BUILD: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
