#!/usr/bin/env python3
"""Serve a local review UI for the 20-sample E4 crop/token PoC."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import mimetypes
import os
from pathlib import Path
import secrets
import sys
import tempfile
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw


STATIC_DIR = Path(__file__).resolve().parents[1] / "web" / "e4-crop-review"
REVIEW_STATUSES = {"pass", "fail", "unsure"}
ISSUE_CODES = {
    "bbox_misaligned",
    "crop_cuts_anomaly",
    "context_too_little",
    "context_too_much",
    "normal_crop_too_easy",
    "orientation_problem",
    "other",
}


class CropReviewError(ValueError):
    """Raised when the PoC manifest or a review value is invalid."""


def load_manifest(path: Path) -> list[dict]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CropReviewError(f"cannot read manifest: {path}") from exc
    if not isinstance(value, dict):
        raise CropReviewError("manifest must be an object")
    if value.get("source_scope") != "broad_clean_train_only":
        raise CropReviewError("manifest is not Train-only")
    if value.get("test_untouched") is not True or value.get("dev_untouched") is not True:
        raise CropReviewError("manifest isolation flags are invalid")
    samples = value.get("samples")
    if not isinstance(samples, list) or not samples:
        raise CropReviewError("manifest samples must be a non-empty list")
    records: list[dict] = []
    for expected_index, sample in enumerate(samples):
        if not isinstance(sample, dict) or sample.get("index") != expected_index:
            raise CropReviewError(f"invalid sample index: {expected_index}")
        if sample.get("sample_type") not in {"T2_BAD", "T3_GOOD"}:
            raise CropReviewError(f"invalid sample type: {expected_index}")
        for field in ("source_image", "crop_image", "image_key", "selection_reason"):
            if not isinstance(sample.get(field), str) or not sample[field]:
                raise CropReviewError(f"invalid {field}: {expected_index}")
        for field in ("source_image", "crop_image"):
            if not Path(sample[field]).is_file():
                raise CropReviewError(f"missing {field}: {sample[field]}")
        crop_box = sample.get("crop_box")
        if (
            not isinstance(crop_box, list)
            or len(crop_box) != 4
            or any(not isinstance(item, int) for item in crop_box)
        ):
            raise CropReviewError(f"invalid crop_box: {expected_index}")
        bbox = sample.get("bbox")
        if sample["sample_type"] == "T2_BAD":
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise CropReviewError(f"missing bbox: {expected_index}")
        elif bbox is not None:
            raise CropReviewError(f"GOOD sample must not have bbox: {expected_index}")
        payload = sample.get("payload")
        if not isinstance(payload, dict) or payload.get("decision") not in {"GOOD", "BAD"}:
            raise CropReviewError(f"invalid payload: {expected_index}")
        records.append(sample)
    return records


def validate_review(value: object) -> dict:
    if not isinstance(value, dict):
        raise CropReviewError("review must be an object")
    status = value.get("status", "")
    if not isinstance(status, str) or (status and status not in REVIEW_STATUSES):
        raise CropReviewError("invalid status")
    issues = value.get("issues", [])
    if not isinstance(issues, list) or any(item not in ISSUE_CODES for item in issues):
        raise CropReviewError("invalid issues")
    issues = list(dict.fromkeys(issues))
    notes = value.get("notes", "")
    if not isinstance(notes, str):
        raise CropReviewError("notes must be a string")
    notes = notes.strip()
    if len(notes) > 2000:
        raise CropReviewError("notes exceeds 2000 characters")
    if status == "pass" and issues:
        raise CropReviewError("PASS review cannot contain issue codes")
    if status == "fail" and not issues:
        raise CropReviewError("FAIL review requires at least one issue")
    return {
        "status": status,
        "issues": issues,
        "notes": notes,
        "completed": status in REVIEW_STATUSES,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


class CropReviewStore:
    def __init__(self, records: list[dict], annotation_path: Path, export_path: Path):
        self.records = records
        self.annotation_path = annotation_path
        self.export_path = export_path
        self.lock = threading.Lock()
        self.annotations = self._load()

    def _load(self) -> dict[str, dict]:
        if not self.annotation_path.exists():
            return {}
        try:
            value = json.loads(self.annotation_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CropReviewError(f"cannot read annotations: {self.annotation_path}") from exc
        if not isinstance(value, dict):
            raise CropReviewError("annotations must be an object")
        result: dict[str, dict] = {}
        for key, review in value.items():
            try:
                index = int(key)
            except ValueError as exc:
                raise CropReviewError(f"invalid annotation index: {key}") from exc
            if index < 0 or index >= len(self.records):
                raise CropReviewError(f"annotation index out of range: {key}")
            normalized = validate_review(review)
            if isinstance(review.get("updated_at"), str):
                normalized["updated_at"] = review["updated_at"]
            result[key] = normalized
        return result

    def snapshot(self) -> dict[str, dict]:
        with self.lock:
            return json.loads(json.dumps(self.annotations, ensure_ascii=False))

    def save(self, index: int, value: object) -> dict:
        if index < 0 or index >= len(self.records):
            raise CropReviewError(f"unknown sample index: {index}")
        review = validate_review(value)
        with self.lock:
            self.annotations[str(index)] = review
            self._persist()
        return review

    def _persist(self) -> None:
        self.annotation_path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.annotations, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.annotation_path.name}.",
            suffix=".tmp",
            dir=self.annotation_path.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self.annotation_path)
            self.export_path.write_bytes(self.export_csv())
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise

    def export_csv(self) -> bytes:
        fields = [
            "index",
            "sample_type",
            "image_key",
            "decision",
            "categories",
            "reasons",
            "bbox_area_ratio",
            "crop_scale",
            "selection_reason",
            "status",
            "issues",
            "notes",
            "completed",
            "updated_at",
        ]
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in self.records:
            review = self.annotations.get(str(record["index"]), {})
            payload = record["payload"]
            writer.writerow(
                {
                    "index": record["index"],
                    "sample_type": record["sample_type"],
                    "image_key": record["image_key"],
                    "decision": payload.get("decision", ""),
                    "categories": " | ".join(payload.get("categories", [])),
                    "reasons": " | ".join(payload.get("reasons", [])),
                    "bbox_area_ratio": record.get("bbox_area_ratio"),
                    "crop_scale": record.get("crop_scale"),
                    "selection_reason": record["selection_reason"],
                    "status": review.get("status", ""),
                    "issues": " | ".join(review.get("issues", [])),
                    "notes": review.get("notes", ""),
                    "completed": review.get("completed", False),
                    "updated_at": review.get("updated_at", ""),
                }
            )
        return ("\ufeff" + stream.getvalue()).encode("utf-8")


def annotated_image_bytes(record: dict, max_side: int = 1600) -> bytes:
    with Image.open(record["source_image"]) as source:
        image = source.convert("RGB")
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale < 1:
        image = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
    draw = ImageDraw.Draw(image)
    line_width = max(3, round(min(image.size) / 250))

    def scaled(box):
        return tuple(round(float(value) * scale) for value in box)

    if record.get("bbox") is not None:
        draw.rectangle(scaled(record["bbox"]), outline=(239, 68, 68), width=line_width)
    draw.rectangle(scaled(record["crop_box"]), outline=(34, 211, 238), width=line_width)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=91, optimize=True)
    return buffer.getvalue()


class CropReviewApplication:
    def __init__(self, records: list[dict], store: CropReviewStore, token: str):
        self.records = records
        self.store = store
        self.token = token

    def state(self) -> dict:
        annotations = self.store.snapshot()
        return {
            "protocol_version": "e4_crop_visual_review_web_v1",
            "total": len(self.records),
            "completed": sum(item.get("completed") is True for item in annotations.values()),
            "records": self.records,
            "annotations": annotations,
        }


def make_handler(app: CropReviewApplication):
    class Handler(BaseHTTPRequestHandler):
        server_version = "E4CropReview/1.0"

        def _authorized(self, parsed) -> bool:
            query_token = parse_qs(parsed.query).get("token", [""])[0]
            header_token = self.headers.get("X-Review-Token", "")
            return secrets.compare_digest(app.token, query_token or header_token)

        def _send(self, status: HTTPStatus, payload: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, status: HTTPStatus, value: object) -> None:
            self._send(
                status,
                json.dumps(value, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def _index(self, parsed, prefix: str) -> int:
            try:
                index = int(parsed.path.removeprefix(prefix))
            except ValueError as exc:
                raise CropReviewError("invalid sample index") from exc
            if index < 0 or index >= len(app.records):
                raise CropReviewError("sample index out of range")
            return index

        def _file(self, path: Path) -> None:
            try:
                payload = path.read_bytes()
            except OSError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "file not found"})
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self._send(HTTPStatus.OK, payload, content_type)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._file(STATIC_DIR / "index.html")
                return
            if parsed.path in {"/app.css", "/app.js"}:
                self._file(STATIC_DIR / parsed.path[1:])
                return
            if not self._authorized(parsed):
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            try:
                if parsed.path == "/api/state":
                    self._json(HTTPStatus.OK, app.state())
                elif parsed.path == "/api/export.csv":
                    self._send(HTTPStatus.OK, app.store.export_csv(), "text/csv; charset=utf-8")
                elif parsed.path.startswith("/api/annotated/"):
                    index = self._index(parsed, "/api/annotated/")
                    self._send(
                        HTTPStatus.OK,
                        annotated_image_bytes(app.records[index]),
                        "image/jpeg",
                    )
                elif parsed.path.startswith("/api/crop/"):
                    index = self._index(parsed, "/api/crop/")
                    self._file(Path(app.records[index]["crop_image"]))
                elif parsed.path.startswith("/api/original/"):
                    index = self._index(parsed, "/api/original/")
                    self._file(Path(app.records[index]["source_image"]))
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except CropReviewError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if not self._authorized(parsed):
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            if not parsed.path.startswith("/api/review/"):
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                index = self._index(parsed, "/api/review/")
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 10000:
                    raise CropReviewError("invalid request length")
                value = json.loads(self.rfile.read(length))
                review = app.store.save(index, value)
                self._json(HTTPStatus.OK, {"review": review})
            except (CropReviewError, json.JSONDecodeError, ValueError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8877)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        records = load_manifest(args.manifest)
        store = CropReviewStore(
            records,
            args.output_dir / "annotations.json",
            args.output_dir / "reviewed.csv",
        )
    except CropReviewError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    token = secrets.token_urlsafe(24)
    app = CropReviewApplication(records, store, token)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"records={len(records)} existing_reviews={len(store.annotations)}", flush=True)
    print(f"URL=http://{args.host}:{args.port}/?token={token}", flush=True)
    print(f"output={args.output_dir}", flush=True)
    print("E4_CROP_REVIEW_WEB: READY", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
