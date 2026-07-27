#!/usr/bin/env python3
"""Serve a blind browser UI for the model-informed Test250 priority review."""

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


STATIC_DIR = Path(__file__).resolve().parents[1] / "web" / "final-test250-review"
DECISIONS = {"GOOD", "BAD", "UNSURE"}
SEVERITIES = {"obvious", "borderline", "none", "uncertain"}
CATEGORIES = {
    "手部异常",
    "文字/符号异常",
    "面部/五官异常",
    "人体结构异常",
    "常识不合理",
    "关系/逻辑矛盾",
    "其他",
}


class BlindReviewError(ValueError):
    """Raised when the blind manifest or an annotation is invalid."""


def load_review_rows(path: Path, expected_count: int = 73) -> list[dict]:
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise BlindReviewError(f"cannot read review manifest: {path}") from exc
    rows: list[dict] = []
    seen: set[int] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BlindReviewError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(record, dict) or set(record) != {
            "review_order",
            "row",
            "index",
            "image_path",
        }:
            raise BlindReviewError(
                f"review row {line_number} contains unexpected fields"
            )
        row = record["row"]
        if not isinstance(row, int) or row < 1 or row in seen:
            raise BlindReviewError(f"invalid or duplicate row at {path}:{line_number}")
        if record["index"] != row - 1:
            raise BlindReviewError(f"index mismatch at Test row {row}")
        if record["review_order"] != len(rows) + 1:
            raise BlindReviewError(f"review order mismatch at Test row {row}")
        if not isinstance(record["image_path"], str) or not record["image_path"]:
            raise BlindReviewError(f"missing image path at Test row {row}")
        seen.add(row)
        rows.append(record)
    if len(rows) != expected_count:
        raise BlindReviewError(
            f"expected {expected_count} review rows, got {len(rows)}"
        )
    return rows


def _clean_text(value: object, field: str, max_length: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise BlindReviewError(f"{field} must be a string")
    cleaned = value.strip()
    if len(cleaned) > max_length:
        raise BlindReviewError(f"{field} exceeds {max_length} characters")
    return cleaned


def validate_annotation(value: object) -> dict:
    if not isinstance(value, dict):
        raise BlindReviewError("annotation must be an object")
    decision = _clean_text(value.get("review_decision"), "review_decision", 16)
    severity = _clean_text(value.get("visible_severity"), "visible_severity", 32)
    if decision and decision not in DECISIONS:
        raise BlindReviewError("invalid review_decision")
    if severity and severity not in SEVERITIES:
        raise BlindReviewError("invalid visible_severity")
    categories = value.get("categories") or []
    if not isinstance(categories, list) or any(item not in CATEGORIES for item in categories):
        raise BlindReviewError("invalid categories")
    if len(categories) != len(set(categories)):
        raise BlindReviewError("duplicate categories")
    notes = _clean_text(value.get("notes"), "notes", 2000)
    return {
        "review_decision": decision,
        "visible_severity": severity,
        "categories": categories,
        "notes": notes,
        "completed": bool(decision and severity),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


class AnnotationStore:
    def __init__(self, records: list[dict], annotation_path: Path, export_path: Path):
        self.records = records
        self.by_row = {record["row"]: record for record in records}
        self.annotation_path = Path(annotation_path)
        self.export_path = Path(export_path)
        self.lock = threading.Lock()
        self.annotations = self._load()

    def _load(self) -> dict[str, dict]:
        if not self.annotation_path.exists():
            return {}
        try:
            value = json.loads(self.annotation_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BlindReviewError(
                f"cannot load annotations: {self.annotation_path}"
            ) from exc
        if not isinstance(value, dict):
            raise BlindReviewError("annotations file must be an object")
        loaded: dict[str, dict] = {}
        for key, annotation in value.items():
            try:
                row = int(key)
            except ValueError as exc:
                raise BlindReviewError(f"invalid annotation row: {key}") from exc
            if row not in self.by_row:
                raise BlindReviewError(f"annotation does not match review row: {key}")
            normalized = validate_annotation(annotation)
            if isinstance(annotation.get("updated_at"), str):
                normalized["updated_at"] = annotation["updated_at"]
            loaded[str(row)] = normalized
        return loaded

    def snapshot(self) -> dict[str, dict]:
        with self.lock:
            return json.loads(json.dumps(self.annotations, ensure_ascii=False))

    def save(self, row: int, value: object) -> dict:
        if row not in self.by_row:
            raise BlindReviewError(f"unknown review row: {row}")
        annotation = validate_annotation(value)
        with self.lock:
            self.annotations[str(row)] = annotation
            self._persist_locked()
        return annotation

    def _persist_locked(self) -> None:
        self.annotation_path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(
            self.annotations, ensure_ascii=False, sort_keys=True, indent=2
        ) + "\n"
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
            self.export_path.write_bytes(self.export_csv_bytes())
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise

    def export_csv_bytes(self) -> bytes:
        fields = [
            "review_order",
            "row",
            "image_path",
            "review_decision",
            "visible_severity",
            "categories",
            "notes",
            "completed",
            "updated_at",
        ]
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in self.records:
            annotation = self.annotations.get(str(record["row"]), {})
            writer.writerow(
                {
                    "review_order": record["review_order"],
                    "row": record["row"],
                    "image_path": record["image_path"],
                    "review_decision": annotation.get("review_decision", ""),
                    "visible_severity": annotation.get("visible_severity", ""),
                    "categories": " | ".join(annotation.get("categories", [])),
                    "notes": annotation.get("notes", ""),
                    "completed": annotation.get("completed", ""),
                    "updated_at": annotation.get("updated_at", ""),
                }
            )
        return ("\ufeff" + stream.getvalue()).encode("utf-8")


class ReviewApplication:
    def __init__(self, records: list[dict], store: AnnotationStore, token: str):
        self.records = records
        self.store = store
        self.token = token
        self.by_row = {record["row"]: record for record in records}

    def state(self) -> dict:
        annotations = self.store.snapshot()
        return {
            "protocol_version": "final_test250_priority_blind_review_web_v1",
            "total": len(self.records),
            "completed": sum(
                annotation.get("completed") is True
                for annotation in annotations.values()
            ),
            "records": self.records,
            "annotations": annotations,
            "original_labels_exposed": False,
            "model_predictions_exposed": False,
        }


def make_handler(app: ReviewApplication):
    class Handler(BaseHTTPRequestHandler):
        server_version = "FinalTest250BlindReview/1.0"

        def _authorized(self, parsed) -> bool:
            query_token = parse_qs(parsed.query).get("token", [""])[0]
            header_token = self.headers.get("X-Audit-Token", "")
            return secrets.compare_digest(app.token, query_token or header_token)

        def _json(self, status: HTTPStatus, value: object) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _bytes(self, status: HTTPStatus, content_type: str, payload: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def _static(self, name: str, content_type: str) -> None:
            try:
                payload = (STATIC_DIR / name).read_bytes()
            except OSError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "static asset missing"})
                return
            self._bytes(HTTPStatus.OK, content_type, payload)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            static = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/app.css": ("app.css", "text/css; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            }
            if parsed.path in static:
                self._static(*static[parsed.path])
                return
            if parsed.path == "/health":
                self._json(HTTPStatus.OK, {"status": "ok"})
                return
            if not self._authorized(parsed):
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "invalid access token"})
                return
            if parsed.path == "/api/state":
                self._json(HTTPStatus.OK, app.state())
                return
            if parsed.path == "/api/export.csv":
                self._bytes(
                    HTTPStatus.OK,
                    "text/csv; charset=utf-8",
                    app.store.export_csv_bytes(),
                )
                return
            if parsed.path.startswith("/api/image/"):
                try:
                    row = int(parsed.path.rsplit("/", 1)[-1])
                    image_path = Path(app.by_row[row]["image_path"])
                    payload = image_path.read_bytes()
                except (ValueError, KeyError, OSError):
                    self._json(HTTPStatus.NOT_FOUND, {"error": "image not available"})
                    return
                content_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "private, max-age=3600")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(payload)
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if not self._authorized(parsed):
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "invalid access token"})
                return
            if not parsed.path.startswith("/api/annotation/"):
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                row = int(parsed.path.rsplit("/", 1)[-1])
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0 or length > 20_000:
                    raise BlindReviewError("request body is too large")
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                annotation = app.store.save(row, body)
            except (
                ValueError,
                json.JSONDecodeError,
                UnicodeDecodeError,
                BlindReviewError,
            ) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._json(HTTPStatus.OK, {"row": row, "annotation": annotation})

        def log_message(self, format: str, *args) -> None:
            print(f"[web] {self.address_string()} {format % args}", file=sys.stderr)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token")
    parser.add_argument("--expected-count", type=int, default=73)
    args = parser.parse_args()
    try:
        if not STATIC_DIR.is_dir():
            raise BlindReviewError(f"web assets do not exist: {STATIC_DIR}")
        records = load_review_rows(
            args.data_dir / "review.jsonl", expected_count=args.expected_count
        )
        token = args.token or secrets.token_urlsafe(18)
        store = AnnotationStore(
            records,
            args.data_dir / "annotations.json",
            args.data_dir / "reviewed.csv",
        )
        app = ReviewApplication(records, store, token)
        server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    except (BlindReviewError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"FINAL_TEST250_PRIORITY_REVIEW_WEB: READY rows={len(records)}")
    print(f"Open: http://{args.host}:{args.port}/?token={token}")
    print(f"Annotations: {store.annotation_path}")
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nFINAL_TEST250_PRIORITY_REVIEW_WEB: STOPPED")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
