#!/usr/bin/env python3
"""Inspect the final Test workbook and image directory without modifying them."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import posixpath
from pathlib import Path, PurePosixPath
import re
from typing import Any
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
DOCUMENT_REL_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CELL_REFERENCE_RE = re.compile(r"^([A-Z]+)([1-9][0-9]*)$")


class SourceAuditError(ValueError):
    """Raised when the final Test sources cannot be inspected safely."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_basename(value: Any) -> str:
    text = str(value).strip().replace("\\", "/")
    return PurePosixPath(text).name.casefold()


def inspect_images(image_root: Path) -> tuple[dict, set[str], set[str]]:
    if not image_root.is_dir():
        raise SourceAuditError(f"image directory does not exist: {image_root}")
    images = sorted(
        (
            path
            for path in image_root.rglob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.relative_to(image_root).as_posix().casefold(),
    )
    if not images:
        raise SourceAuditError(f"no supported images under: {image_root}")

    names = Counter(path.name.casefold() for path in images)
    stems = Counter(path.stem.casefold() for path in images)
    extensions = Counter(path.suffix.casefold() for path in images)
    unreadable: list[str] = []
    manifest_digest = hashlib.sha256()
    for path in images:
        relative = path.relative_to(image_root).as_posix()
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError) as exc:
            unreadable.append(f"{relative}: {exc}")
        digest = file_sha256(path)
        manifest_digest.update(relative.encode("utf-8"))
        manifest_digest.update(b"\0")
        manifest_digest.update(str(path.stat().st_size).encode("ascii"))
        manifest_digest.update(b"\0")
        manifest_digest.update(digest.encode("ascii"))
        manifest_digest.update(b"\n")

    return (
        {
            "root": str(image_root.resolve()),
            "count": len(images),
            "extension_counts": dict(sorted(extensions.items())),
            "duplicate_basenames": sorted(name for name, count in names.items() if count > 1),
            "duplicate_stems": sorted(name for name, count in stems.items() if count > 1),
            "unreadable_count": len(unreadable),
            "unreadable_examples": unreadable[:10],
            "manifest_sha256": manifest_digest.hexdigest(),
        },
        set(names),
        set(stems),
    )


def compact_value(value: Any, limit: int = 120) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = str(value).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def column_number(letters: str) -> int:
    value = 0
    for letter in letters:
        value = value * 26 + ord(letter) - ord("A") + 1
    return value


def column_letter(number: int) -> str:
    if number < 1:
        raise SourceAuditError(f"invalid column number: {number}")
    letters = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


def parse_cell_reference(reference: str) -> tuple[int, int]:
    match = CELL_REFERENCE_RE.fullmatch(reference)
    if match is None:
        raise SourceAuditError(f"invalid XLSX cell reference: {reference}")
    return int(match.group(2)), column_number(match.group(1))


def xml_text(element: ElementTree.Element | None) -> str:
    if element is None:
        return ""
    return "".join(node.text or "" for node in element.iter(f"{{{MAIN_NS}}}t"))


def parse_shared_strings(archive: ZipFile) -> list[str]:
    path = "xl/sharedStrings.xml"
    if path not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read(path))
    return [xml_text(item) for item in root.findall(f"{{{MAIN_NS}}}si")]


def parse_cell_value(
    cell: ElementTree.Element,
    shared_strings: list[str],
) -> Any:
    cell_type = cell.get("t")
    if cell_type == "inlineStr":
        return xml_text(cell.find(f"{{{MAIN_NS}}}is"))
    value_node = cell.find(f"{{{MAIN_NS}}}v")
    if value_node is None or value_node.text is None:
        return None
    raw = value_node.text
    if cell_type == "s":
        try:
            return shared_strings[int(raw)]
        except (ValueError, IndexError) as exc:
            raise SourceAuditError(f"invalid shared-string index: {raw}") from exc
    if cell_type == "b":
        return raw == "1"
    if cell_type in {"str", "e", "d"}:
        return raw
    try:
        numeric = float(raw)
    except ValueError:
        return raw
    return int(numeric) if numeric.is_integer() else numeric


def workbook_sheet_paths(archive: ZipFile) -> list[dict]:
    workbook_path = "xl/workbook.xml"
    relationship_path = "xl/_rels/workbook.xml.rels"
    try:
        workbook_root = ElementTree.fromstring(archive.read(workbook_path))
        relationship_root = ElementTree.fromstring(archive.read(relationship_path))
    except KeyError as exc:
        raise SourceAuditError(f"XLSX is missing required part: {exc}") from exc
    targets = {
        relationship.get("Id"): relationship.get("Target")
        for relationship in relationship_root.findall(
            f"{{{PACKAGE_REL_NS}}}Relationship"
        )
    }
    sheets: list[dict] = []
    for sheet in workbook_root.findall(f".//{{{MAIN_NS}}}sheet"):
        relationship_id = sheet.get(f"{{{DOCUMENT_REL_NS}}}id")
        target = targets.get(relationship_id)
        if not target:
            raise SourceAuditError(
                f"XLSX sheet {sheet.get('name')} lacks a worksheet relationship"
            )
        if target.startswith("/"):
            resolved = target.lstrip("/")
        else:
            resolved = posixpath.normpath(
                posixpath.join(posixpath.dirname(workbook_path), target)
            )
        sheets.append(
            {
                "name": sheet.get("name", ""),
                "state": sheet.get("state", "visible"),
                "path": resolved,
            }
        )
    return sheets


def inspect_sheet_xml(
    archive: ZipFile,
    sheet_meta: dict,
    shared_strings: list[str],
    image_names: set[str],
    image_stems: set[str],
) -> dict:
    try:
        root = ElementTree.fromstring(archive.read(sheet_meta["path"]))
    except KeyError as exc:
        raise SourceAuditError(
            f"XLSX sheet part does not exist: {sheet_meta['path']}"
        ) from exc
    preview: list[dict] = []
    formula_count = 0
    nonempty_count = 0
    max_row = 0
    max_column = 0
    column_values: dict[int, list[Any]] = {}
    for cell in root.findall(f".//{{{MAIN_NS}}}c"):
        reference = cell.get("r")
        if not reference:
            continue
        row_number, column = parse_cell_reference(reference)
        max_row = max(max_row, row_number)
        max_column = max(max_column, column)
        if cell.find(f"{{{MAIN_NS}}}f") is not None:
            formula_count += 1
        value = parse_cell_value(cell, shared_strings)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        nonempty_count += 1
        column_values.setdefault(column, []).append(value)
        if len(preview) < 30 and row_number <= 15 and column <= 20:
            preview.append({"cell": reference, "value": compact_value(value)})

    profiles: list[dict] = []
    for column, values in sorted(column_values.items()):
        strings = [str(value).strip() for value in values]
        normalized_names = [normalized_basename(value) for value in values]
        counts = Counter(strings)
        profiles.append(
            {
                "column": column_letter(column),
                "nonempty": len(values),
                "unique": len(counts),
                "top_values": [
                    {"value": compact_value(value, 80), "count": count}
                    for value, count in counts.most_common(8)
                ],
                "image_basename_matches": sum(
                    name in image_names for name in normalized_names
                ),
                "image_stem_matches": sum(
                    PurePosixPath(name).stem.casefold() in image_stems
                    for name in normalized_names
                ),
            }
        )
    merged_ranges = [
        node.get("ref")
        for node in root.findall(f".//{{{MAIN_NS}}}mergeCell")
        if node.get("ref")
    ]
    return {
        "name": sheet_meta["name"],
        "state": sheet_meta["state"],
        "max_row": max_row,
        "max_column": max_column,
        "nonempty_cells": nonempty_count,
        "formula_cells": formula_count,
        "merged_ranges": merged_ranges,
        "preview": preview,
        "column_profiles": profiles,
    }


def inspect_workbook(
    workbook_path: Path,
    image_names: set[str],
    image_stems: set[str],
) -> dict:
    if not workbook_path.is_file():
        raise SourceAuditError(f"workbook does not exist: {workbook_path}")
    try:
        archive = ZipFile(workbook_path)
    except (BadZipFile, OSError) as exc:
        raise SourceAuditError(f"cannot open workbook: {exc}") from exc

    try:
        shared_strings = parse_shared_strings(archive)
        sheets = [
            inspect_sheet_xml(
                archive,
                sheet_meta,
                shared_strings,
                image_names,
                image_stems,
            )
            for sheet_meta in workbook_sheet_paths(archive)
        ]
    finally:
        archive.close()
    return {
        "path": str(workbook_path.resolve()),
        "size_bytes": workbook_path.stat().st_size,
        "sha256": file_sha256(workbook_path),
        "sheet_count": len(sheets),
        "sheets": sheets,
    }


def audit_sources(workbook_path: Path, image_root: Path) -> dict:
    image_summary, image_names, image_stems = inspect_images(image_root)
    workbook_summary = inspect_workbook(workbook_path, image_names, image_stems)
    return {
        "protocol_version": "final_test250_source_inspection_v1",
        "source_only": True,
        "model_inference_run": False,
        "workbook": workbook_summary,
        "images": image_summary,
        "status": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        summary = audit_sources(args.workbook, args.image_root)
    except (SourceAuditError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print("FINAL_TEST250_SOURCE_INSPECTION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
