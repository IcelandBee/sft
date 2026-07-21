#!/usr/bin/env python3
"""Serve a local browser UI for reviewing E1/E2 Dev boundary cases."""

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


DEFAULT_DATA_DIR = Path(
    "/home/data/h30082292/data/pose/artifact_detection_training/evaluations/"
    "e1_e2_dev_boundary_audit_v1"
)
STATIC_DIR = Path(__file__).resolve().parents[1] / "web" / "dev-audit"
LABEL_STATUSES = {"gold_correct", "gold_incorrect", "uncertain"}
SEVERITIES = {"obvious", "borderline", "none", "uncertain"}
REVIEW_DECISIONS = {"GOOD", "BAD", "UNSURE"}


class AuditWebError(ValueError):
    """Raised when review data or an annotation violates the UI contract."""


def load_review_rows(path: Path) -> list[dict]:
    """Load and validate the immutable review manifest."""
    rows: list[dict] = []
    seen: set[int] = set()
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise AuditWebError(f"cannot read review file: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AuditWebError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(record, dict):
            raise AuditWebError(f"review row {line_number} must be an object")
        row = record.get("row")
        image_path = record.get("image_path")
        if not isinstance(row, int) or row < 1 or row in seen:
            raise AuditWebError(f"invalid or duplicate row at {path}:{line_number}")
        if not isinstance(image_path, str) or not image_path:
            raise AuditWebError(f"missing image path at {path}:{line_number}")
        if record.get("review_group") not in {
            "both_wrong",
            "e1_only_correct",
            "e2_only_correct",
        }:
            raise AuditWebError(f"invalid review group at {path}:{line_number}")
        for name in ("gold", "e1", "e2"):
            if not isinstance(record.get(name), dict):
                raise AuditWebError(f"missing {name} payload at {path}:{line_number}")
        seen.add(row)
        rows.append(record)
    if not rows:
        raise AuditWebError(f"review file is empty: {path}")
    return rows


def _clean_text(value: object, field: str, max_length: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise AuditWebError(f"{field} must be a string")
    value = value.strip()
    if len(value) > max_length:
        raise AuditWebError(f"{field} exceeds {max_length} characters")
    return value


def validate_annotation(value: object) -> dict:
    """Normalize a browser annotation and reject unknown values."""
    if not isinstance(value, dict):
        raise AuditWebError("annotation must be an object")
    label_status = _clean_text(value.get("label_status"), "label_status", 32)
    severity = _clean_text(value.get("visible_severity"), "visible_severity", 32)
    decision = _clean_text(value.get("review_decision"), "review_decision", 16)
    if label_status and label_status not in LABEL_STATUSES:
        raise AuditWebError("invalid label_status")
    if severity and severity not in SEVERITIES:
        raise AuditWebError("invalid visible_severity")
    if decision and decision not in REVIEW_DECISIONS:
        raise AuditWebError("invalid review_decision")
    primary_category = _clean_text(
        value.get("primary_category"), "primary_category", 100
    )
    notes = _clean_text(value.get("notes"), "notes", 2000)
    return {
        "label_status": label_status,
        "visible_severity": severity,
        "review_decision": decision,
        "primary_category": primary_category,
        "notes": notes,
        "completed": bool(label_status and severity and decision),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _payload_list(payload: object, field: str) -> str:
    if not isinstance(payload, dict):
        return ""
    values = payload.get(field)
    return " | ".join(str(item) for item in values) if isinstance(values, list) else ""


class AnnotationStore:
    """Thread-safe, atomically persisted annotations for one review manifest."""

    def __init__(self, records: list[dict], annotation_path: Path, export_path: Path):
        self.records = records
        self.by_row = {record["row"]: record for record in records}
        self.annotation_path = Path(annotation_path)
        self.export_path = Path(export_path)
        self.lock = threading.Lock()
        self.annotations = self._load_annotations()

    def _load_annotations(self) -> dict[str, dict]:
        if not self.annotation_path.exists():
            return {}
        try:
            value = json.loads(self.annotation_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AuditWebError(
                f"cannot load annotations: {self.annotation_path}"
            ) from exc
        if not isinstance(value, dict):
            raise AuditWebError("annotations file must be an object")
        loaded: dict[str, dict] = {}
        for key, annotation in value.items():
            try:
                row = int(key)
            except ValueError as exc:
                raise AuditWebError(f"invalid annotation row: {key}") from exc
            if row not in self.by_row or not isinstance(annotation, dict):
                raise AuditWebError(f"annotation does not match review row: {key}")
            # Keep original timestamps while validating all user-editable fields.
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
            raise AuditWebError(f"unknown review row: {row}")
        normalized = validate_annotation(value)
        with self.lock:
            self.annotations[str(row)] = normalized
            self._persist_locked()
        return normalized

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
            "row",
            "review_group",
            "image_path",
            "gold_decision",
            "gold_categories",
            "gold_reasons",
            "e1_decision",
            "e1_categories",
            "e1_reasons",
            "e2_decision",
            "e2_categories",
            "e2_reasons",
            "label_status",
            "visible_severity",
            "review_decision",
            "primary_category",
            "notes",
            "completed",
            "updated_at",
        ]
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in self.records:
            annotation = self.annotations.get(str(record["row"]), {})
            gold = record["gold"]
            e1 = record["e1"].get("payload")
            e2 = record["e2"].get("payload")
            writer.writerow(
                {
                    "row": record["row"],
                    "review_group": record["review_group"],
                    "image_path": record["image_path"],
                    "gold_decision": gold.get("decision", ""),
                    "gold_categories": _payload_list(gold, "categories"),
                    "gold_reasons": _payload_list(gold, "reasons"),
                    "e1_decision": record["e1"].get("decision") or "INVALID",
                    "e1_categories": _payload_list(e1, "categories"),
                    "e1_reasons": _payload_list(e1, "reasons"),
                    "e2_decision": record["e2"].get("decision") or "INVALID",
                    "e2_categories": _payload_list(e2, "categories"),
                    "e2_reasons": _payload_list(e2, "reasons"),
                    **{field: annotation.get(field, "") for field in fields[12:]},
                }
            )
        return ("\ufeff" + stream.getvalue()).encode("utf-8")


class AuditApplication:
    def __init__(self, records: list[dict], store: AnnotationStore, token: str):
        self.records = records
        self.store = store
        self.token = token
        self.by_row = {record["row"]: record for record in records}

    def state(self) -> dict:
        annotations = self.store.snapshot()
        completed = sum(item.get("completed") is True for item in annotations.values())
        return {
            "protocol_version": "e1_e2_dev_review_web_v1",
            "total": len(self.records),
            "completed": completed,
            "records": self.records,
            "annotations": annotations,
        }


def make_handler(app: AuditApplication):
    class Handler(BaseHTTPRequestHandler):
        server_version = "DevAudit/1.0"

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
            path = STATIC_DIR / name
            try:
                payload = path.read_bytes()
            except OSError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "static asset missing"})
                return
            self._bytes(HTTPStatus.OK, content_type, payload)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._static("index.html", "text/html; charset=utf-8")
                return
            if parsed.path == "/app.css":
                self._static("app.css", "text/css; charset=utf-8")
                return
            if parsed.path == "/app.js":
                self._static("app.js", "text/javascript; charset=utf-8")
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
                    raise AuditWebError("request body is too large")
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                annotation = app.store.save(row, body)
            except (ValueError, json.JSONDecodeError, UnicodeDecodeError, AuditWebError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._json(HTTPStatus.OK, {"row": row, "annotation": annotation})

        def log_message(self, format: str, *args) -> None:
            print(f"[web] {self.address_string()} {format % args}", file=sys.stderr)

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not STATIC_DIR.is_dir():
            raise AuditWebError(f"web assets do not exist: {STATIC_DIR}")
        records = load_review_rows(args.data_dir / "review.jsonl")
        token = args.token or secrets.token_urlsafe(18)
        store = AnnotationStore(
            records,
            args.data_dir / "annotations.json",
            args.data_dir / "reviewed.csv",
        )
        app = AuditApplication(records, store, token)
        server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    except (AuditWebError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"DEV_AUDIT_WEB: READY rows={len(records)}")
    print(f"Open: http://{args.host}:{args.port}/?token={token}")
    print(f"Annotations: {store.annotation_path}")
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDEV_AUDIT_WEB: STOPPED")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
