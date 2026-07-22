#!/usr/bin/env python3
"""Run a Transformers-only GOOD/BAD token-trie protocol on a fixed Dev JSONL."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Iterable, Sequence


NON_THINKING_PREFIX = "<think>\n\n</think>\n\n"
GOOD_PAYLOAD = '{"decision":"GOOD","categories":[],"reasons":[]}'
BAD_PAYLOAD = '{"decision":"BAD","categories":["异常"],"reasons":["存在明显生成异常"]}'
PAYLOAD_TO_DECISION = {GOOD_PAYLOAD: "GOOD", BAD_PAYLOAD: "BAD"}


class ConstrainedBinaryError(ValueError):
    """Raised when data, token constraints, or engine output violate the protocol."""


class BinaryTokenTrie:
    """Allow exactly one of a finite set of token sequences, followed by EOS."""

    def __init__(self, candidates: Sequence[Sequence[int]], eos_token_id: int):
        normalized = tuple(tuple(int(token) for token in candidate) for candidate in candidates)
        if len(normalized) < 2 or any(not candidate for candidate in normalized):
            raise ConstrainedBinaryError("at least two non-empty token candidates are required")
        if len(set(normalized)) != len(normalized):
            raise ConstrainedBinaryError("token candidates must be unique")
        for index, candidate in enumerate(normalized):
            for other_index, other in enumerate(normalized):
                if index != other_index and len(candidate) <= len(other) and other[:len(candidate)] == candidate:
                    raise ConstrainedBinaryError("one token candidate cannot prefix another")
        self.candidates = normalized
        self.eos_token_id = int(eos_token_id)
        self.prompt_length: int | None = None

    def bind_prompt_length(self, prompt_length: int) -> None:
        if isinstance(prompt_length, bool) or not isinstance(prompt_length, int) or prompt_length <= 0:
            raise ConstrainedBinaryError("prompt length must be a positive integer")
        self.prompt_length = prompt_length

    def next_tokens(self, generated: Sequence[int]) -> list[int]:
        prefix = tuple(int(token) for token in generated)
        matches = [candidate for candidate in self.candidates if candidate[:len(prefix)] == prefix]
        if not matches:
            raise ConstrainedBinaryError(f"generated tokens left the candidate trie: {prefix}")
        if any(len(candidate) == len(prefix) for candidate in matches):
            if not all(len(candidate) == len(prefix) for candidate in matches):
                raise ConstrainedBinaryError("ambiguous completed token candidate")
            return [self.eos_token_id]
        return sorted({candidate[len(prefix)] for candidate in matches})

    def prefix_allowed_tokens_fn(self, batch_id, input_ids) -> list[int]:
        del batch_id
        if self.prompt_length is None:
            raise ConstrainedBinaryError("prompt length was not bound before generation")
        token_ids = input_ids.tolist()
        if len(token_ids) < self.prompt_length:
            raise ConstrainedBinaryError("generation input is shorter than the bound prompt")
        return self.next_tokens(token_ids[self.prompt_length:])


def _load_dev(path: Path, expected_sha256: str) -> tuple[list[dict], str]:
    path = Path(path)
    try:
        source = path.read_bytes()
    except OSError as exc:
        raise ConstrainedBinaryError(f"cannot read Dev: {path}") from exc
    digest = hashlib.sha256(source).hexdigest()
    if digest != expected_sha256:
        raise ConstrainedBinaryError(f"Dev sha256 mismatch: {digest} != {expected_sha256}")
    rows: list[dict] = []
    for line_number, line in enumerate(source.decode("utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConstrainedBinaryError(f"invalid Dev JSON at line {line_number}") from exc
        if not isinstance(row, dict):
            raise ConstrainedBinaryError(f"Dev row {line_number} must be an object")
        rows.append(row)
    if len(rows) != 200:
        raise ConstrainedBinaryError(f"expected 200 Dev rows, got {len(rows)}")

    decisions: Counter[str] = Counter()
    seen_images: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        if set(row) != {"images", "messages"}:
            raise ConstrainedBinaryError(f"invalid Dev fields at row {row_number}")
        images = row["images"]
        messages = row["messages"]
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
            raise ConstrainedBinaryError(f"invalid image at Dev row {row_number}")
        if images[0] in seen_images:
            raise ConstrainedBinaryError(f"duplicate image at Dev row {row_number}")
        seen_images.add(images[0])
        if not Path(images[0]).is_file():
            raise ConstrainedBinaryError(f"missing image at Dev row {row_number}: {images[0]}")
        if (
            not isinstance(messages, list)
            or len(messages) != 3
            or [message.get("role") for message in messages] != ["system", "user", "assistant"]
        ):
            raise ConstrainedBinaryError(f"invalid messages at Dev row {row_number}")
        try:
            gold = json.loads(messages[-1]["content"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ConstrainedBinaryError(f"invalid gold at Dev row {row_number}") from exc
        decision = gold.get("decision") if isinstance(gold, dict) else None
        if decision not in {"GOOD", "BAD"}:
            raise ConstrainedBinaryError(f"invalid gold decision at Dev row {row_number}")
        decisions[decision] += 1
    if decisions != {"GOOD": 142, "BAD": 58}:
        raise ConstrainedBinaryError(f"Dev label counts mismatch: {dict(decisions)}")
    return rows, digest


def _tokenize_candidates(tokenizer) -> tuple[list[list[int]], dict[tuple[int, ...], str]]:
    candidate_ids: list[list[int]] = []
    ids_to_payload: dict[tuple[int, ...], str] = {}
    for payload in (GOOD_PAYLOAD, BAD_PAYLOAD):
        token_ids = tokenizer.encode(payload, add_special_tokens=False)
        decoded = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded != payload:
            raise ConstrainedBinaryError(
                f"candidate tokenizer round-trip failed: {payload!r} != {decoded!r}"
            )
        candidate_ids.append(token_ids)
        ids_to_payload[tuple(token_ids)] = payload
    return candidate_ids, ids_to_payload


def _install_constraint(engine, trie: BinaryTokenTrie) -> None:
    original_prepare = engine.template.prepare_generate_kwargs

    def prepare_generate_kwargs(generate_kwargs, model=None):
        prepared = original_prepare(generate_kwargs, model=model)
        input_ids = prepared.get("input_ids")
        if input_ids is None or getattr(input_ids, "ndim", None) != 2:
            raise ConstrainedBinaryError("prepared generation kwargs lack batched input_ids")
        if input_ids.shape[0] != 1:
            raise ConstrainedBinaryError(
                f"binary trie protocol requires batch size 1, got {input_ids.shape[0]}"
            )
        trie.bind_prompt_length(int(input_ids.shape[1]))
        if "prefix_allowed_tokens_fn" in prepared:
            raise ConstrainedBinaryError("generation kwargs already contain a prefix constraint")
        prepared["prefix_allowed_tokens_fn"] = trie.prefix_allowed_tokens_fn
        return prepared

    engine.template.prepare_generate_kwargs = prepare_generate_kwargs


def _result_row(source: dict, response: str, logprobs) -> dict:
    messages = deepcopy(source["messages"][:2])
    messages.append({"role": "assistant", "content": response})
    return {
        "response": response,
        "labels": source["messages"][-1]["content"],
        "logprobs": logprobs,
        "images": [{"bytes": None, "path": source["images"][0]}],
        "messages": messages,
        "dataset": None,
    }


def _jsonl_text(rows: Iterable[dict]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    )


def run(args: argparse.Namespace) -> dict:
    try:
        import torch
        from swift import TransformersEngine
        from swift.infer_engine import InferRequest, RequestConfig
    except ImportError as exc:
        raise ConstrainedBinaryError(f"required runtime import failed: {exc}") from exc

    dev_rows, dev_sha256 = _load_dev(args.dev, args.expected_dev_sha256)
    adapters = None if args.adapter is None else [str(args.adapter)]
    start = time.monotonic()
    engine = TransformersEngine(
        str(args.model),
        adapters=adapters,
        max_batch_size=1,
        torch_dtype=torch.bfloat16,
        attn_impl="flash_attention_2",
    )
    engine.strict = True
    engine.template.enable_thinking = False
    engine.template.response_prefix = None

    candidate_ids, _ = _tokenize_candidates(engine.tokenizer)
    eos_token_id = engine.tokenizer.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]
    if eos_token_id is None:
        raise ConstrainedBinaryError("tokenizer has no eos_token_id")
    trie = BinaryTokenTrie(candidate_ids, eos_token_id)
    _install_constraint(engine, trie)

    requests = [
        InferRequest(messages=deepcopy(row["messages"][:2]), images=list(row["images"]))
        for row in dev_rows
    ]
    request_config = RequestConfig(
        max_tokens=max(len(candidate) for candidate in candidate_ids) + 1,
        temperature=0.0,
        num_beams=1,
        seed=42,
        stream=False,
        return_details=True,
    )
    responses = engine.infer(requests, request_config=request_config, use_tqdm=True)
    if len(responses) != len(dev_rows):
        raise ConstrainedBinaryError(
            f"response count mismatch: {len(responses)} != {len(dev_rows)}"
        )

    result_rows: list[dict] = []
    decision_counts: Counter[str] = Counter()
    completion_tokens = 0
    for row_number, (source, response) in enumerate(zip(dev_rows, responses), start=1):
        if response is None or len(response.choices) != 1:
            raise ConstrainedBinaryError(f"invalid engine response at row {row_number}")
        choice = response.choices[0]
        text = choice.message.content
        if not isinstance(text, str) or not text.startswith(NON_THINKING_PREFIX):
            raise ConstrainedBinaryError(f"non-canonical response prefix at row {row_number}: {text!r}")
        payload = text[len(NON_THINKING_PREFIX):]
        decision = PAYLOAD_TO_DECISION.get(payload)
        if decision is None:
            raise ConstrainedBinaryError(f"response escaped binary payloads at row {row_number}: {text!r}")
        decision_counts[decision] += 1
        completion_tokens += response.usage.completion_tokens
        result_row = _result_row(source, text, choice.logprobs)
        result_row["dataset"] = str(args.dev)
        result_rows.append(result_row)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    result_path = args.output_dir / "raw-result.jsonl"
    metadata_path = args.output_dir / "protocol-metadata.json"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        delete=False,
        dir=args.output_dir,
        prefix=".raw-result.",
        suffix=".tmp",
    ) as handle:
        handle.write(_jsonl_text(result_rows))
        temp_result = Path(handle.name)
    temp_result.replace(result_path)
    runtime = time.monotonic() - start
    metadata = {
        "protocol_version": "transformers_binary_token_trie_v1",
        "model": str(args.model),
        "adapter": None if args.adapter is None else str(args.adapter),
        "dev": str(args.dev),
        "dev_sha256": dev_sha256,
        "infer_backend": "transformers",
        "temperature": 0.0,
        "num_beams": 1,
        "seed": 42,
        "max_batch_size": 1,
        "candidate_payloads": [GOOD_PAYLOAD, BAD_PAYLOAD],
        "candidate_token_ids": candidate_ids,
        "schema_validity": "guaranteed_by_token_constraint_not_model_capability",
        "decision_counts": dict(sorted(decision_counts.items())),
        "num_samples": len(result_rows),
        "num_completion_tokens": completion_tokens,
        "runtime_seconds": runtime,
        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "test_untouched": True,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--expected-dev-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists():
        print(f"ERROR: output directory already exists: {args.output_dir}", file=sys.stderr)
        return 2
    try:
        metadata = run(args)
    except (ConstrainedBinaryError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("CONSTRAINED_BINARY_INFER: PASS")
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
