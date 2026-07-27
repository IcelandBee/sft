#!/usr/bin/env python3
"""Validate four merged Qwen3.5 Hugging Face model directories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_ARCHITECTURE = ["Qwen3_5ForConditionalGeneration"]
MIN_MERGED_WEIGHT_BYTES = 50_000_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_entry(value: str) -> tuple[str, Path]:
    name, separator, adapter = value.partition("=")
    if not separator or not name or not adapter:
        raise argparse.ArgumentTypeError("entry must be NAME=ADAPTER_PATH")
    return name, Path(adapter)


def validate(root: Path, base_model: Path, entries: list[tuple[str, Path]]) -> dict:
    if len(entries) != 4 or len({name for name, _ in entries}) != 4:
        raise ValueError("exactly four unique model entries are required")
    base_config_path = base_model / "config.json"
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    models = []
    for name, adapter in entries:
        merged = root / name
        adapter_config_path = adapter / "adapter_config.json"
        if not adapter_config_path.is_file():
            raise ValueError(f"adapter config is missing for {name}")
        required_files = (
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "preprocessor_config.json",
            "model.safetensors.index.json",
        )
        for required in required_files:
            if not (merged / required).is_file():
                raise ValueError(f"merged model {name} is missing {required}")

        weights = sorted(merged.glob("model-*.safetensors"))
        if not weights:
            raise ValueError(f"merged model {name} has no sharded safetensors weights")
        if list(merged.glob("adapter_model*.safetensors")):
            raise ValueError(f"merged model {name} still contains adapter weights")

        config_path = merged / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        if config.get("model_type") != "qwen3_5":
            raise ValueError(f"unexpected model_type for {name}: {config.get('model_type')}")
        if config.get("architectures") != EXPECTED_ARCHITECTURE:
            raise ValueError(f"unexpected architecture for {name}: {config.get('architectures')}")
        if config.get("model_type") != base_config.get("model_type"):
            raise ValueError(f"base/merged model_type mismatch for {name}")

        weight_bytes = sum(path.stat().st_size for path in weights)
        if weight_bytes < MIN_MERGED_WEIGHT_BYTES:
            raise ValueError(f"merged model {name} appears truncated: {weight_bytes} bytes")
        models.append(
            {
                "name": name,
                "source_adapter": str(adapter),
                "adapter_rank": adapter_config.get("r"),
                "adapter_alpha": adapter_config.get("lora_alpha"),
                "model_directory": str(merged),
                "model_type": config.get("model_type"),
                "architectures": config.get("architectures"),
                "weight_files": len(weights),
                "weight_bytes": weight_bytes,
                "config_sha256": sha256(config_path),
                "weight_index_sha256": sha256(merged / "model.safetensors.index.json"),
            }
        )

    manifest = {
        "protocol_version": "qwen35_27b_four_merged_hf_models_v1",
        "format": "Hugging Face BF16 full model",
        "base_model": str(base_model),
        "base_config_sha256": sha256(base_config_path),
        "model_count": len(models),
        "models": models,
        "runtime_environment_included": False,
        "archive_included": False,
    }
    (root / "four-merged-models-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--entry", action="append", type=parse_entry, required=True)
    args = parser.parse_args()
    manifest = validate(args.root, args.base_model, args.entry)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    print("FOUR_MERGED_MODELS_VALIDATION: PASS")


if __name__ == "__main__":
    main()
